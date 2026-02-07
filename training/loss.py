# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------
# Ranking Loss Helpers (ported from RankGAN)
# Reference: gan_rank/src/rankgan.py
#----------------------------------------------------------------------------

def listmle_loss(scores_sorted: torch.Tensor) -> torch.Tensor:
    """
    ListMLE loss for learning to rank.
    scores_sorted: [B, K] where each row is ordered best->worst.
    The loss encourages D to assign decreasing scores along the ranking.
    """
    rev = torch.flip(scores_sorted, dims=[-1])
    rev_lse = torch.logcumsumexp(rev, dim=-1)
    suffix_lse = torch.flip(rev_lse, dims=[-1])
    loss = (suffix_lse - scores_sorted).sum(dim=-1)
    return loss.mean()


def pairwise_logistic_loss(scores_sorted: torch.Tensor) -> torch.Tensor:
    """
    Pairwise logistic ranking loss.
    scores_sorted: [B, K] where each row is ordered best->worst.
    Encourages D(x_i) > D(x_j) for all i < j.
    """
    B, K = scores_sorted.shape
    s_i = scores_sorted.unsqueeze(2)  # [B, K, 1]
    s_j = scores_sorted.unsqueeze(1)  # [B, 1, K]
    diff = s_i - s_j  # [B, K, K]
    mask = torch.triu(torch.ones(K, K, device=scores_sorted.device), diagonal=1)
    loss_all = F.softplus(-diff)
    loss = (loss_all * mask).sum() / (B * mask.sum())
    return loss


def pairwise_hinge_loss(scores_sorted: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Pairwise hinge loss: max(0, margin - (s_i - s_j)) for i < j.
    """
    B, K = scores_sorted.shape
    s_i = scores_sorted.unsqueeze(2)
    s_j = scores_sorted.unsqueeze(1)
    diff = s_i - s_j
    mask = torch.triu(torch.ones(K, K, device=scores_sorted.device), diagonal=1)
    loss_all = F.relu(margin - diff)
    loss = (loss_all * mask).sum() / (B * mask.sum())
    return loss


def make_rank_list(real_imgs: torch.Tensor, fake_imgs: torch.Tensor, K: int,
                   mode: str = 'intrpl', alpha_dist: str = 'linear') -> torch.Tensor:
    """
    Create ranking list by interpolating between real (best) and fake (worst).
    Returns: [B, K, C, H, W] tensor where position 0 = real, position K-1 = fake.

    Modes:
      - 'intrpl': Linear interpolation between real and fake.
      - 'noise':  Adding Gaussian noise to real images with increasing intensity.
      - 'add_mix': Interpolation + additive noise.

    Alpha Distributions:
      - 'linear':  Evenly spaced between 1.0 and 0.0.
      - 'cosine':  More samples near 1.0 and 0.0.
      - 'random':  Stochastic alphas, sorted descending.
    """
    device = real_imgs.device

    # Generate alpha distribution
    if alpha_dist == 'linear':
        alphas = torch.linspace(1.0, 0.0, K, device=device)
    elif alpha_dist == 'cosine':
        alphas = 0.5 * (1.0 + torch.cos(torch.linspace(0, np.pi, K, device=device)))
    elif alpha_dist == 'random':
        alphas = torch.cat([
            torch.ones(1, device=device),
            torch.rand(K - 1, device=device)
        ])
        alphas = torch.sort(alphas, descending=True)[0]
    else:
        alphas = torch.linspace(1.0, 0.0, K, device=device)

    alphas = alphas.view(1, K, 1, 1, 1)

    if mode == 'intrpl':
        return alphas * real_imgs.unsqueeze(1) + (1.0 - alphas) * fake_imgs.unsqueeze(1)
    elif mode == 'noise':
        noise = torch.randn_like(real_imgs).unsqueeze(1) * 0.01
        return real_imgs.unsqueeze(1) + (1.0 - alphas) * noise
    elif mode == 'add_mix':
        interp = alphas * real_imgs.unsqueeze(1) + (1.0 - alphas) * fake_imgs.unsqueeze(1)
        noise = torch.randn_like(real_imgs).unsqueeze(1) * 0.01
        return interp + (1.0 - alphas) * noise
    else:
        return alphas * real_imgs.unsqueeze(1) + (1.0 - alphas) * fake_imgs.unsqueeze(1)

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
                 # Rank loss options (ported from RankGAN)
                 # Rank loss is a structured regularizer on D: it requires D to assign
                 # monotonically decreasing scores along a real->fake interpolation list.
                 # The standard adversarial loss (softplus) is ALWAYS kept at full strength;
                 # rank loss is purely additive, controlled by lambda_rank.
                 rank_loss=False, rank_K=8, rank_loss_type='listmle',
                 lambda_rank=0.1, lambda_adv=1.0, rank_mode='intrpl',
                 rank_alpha_dist='linear', rank_augment=False, rank_margin=1.0):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

        # Rank loss options
        self.rank_loss = rank_loss
        self.rank_K = rank_K
        self.rank_loss_type = rank_loss_type
        self.lambda_rank = lambda_rank
        self.lambda_adv = lambda_adv
        self.rank_mode = rank_mode
        self.rank_alpha_dist = rank_alpha_dist
        self.rank_augment = rank_augment
        self.rank_margin = rank_margin

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync, augment=True):
        if augment and self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain * self.lambda_adv).backward()

        # Drank: Ranking loss as structured regularizer on D.
        # Constructs a ranked list [real, interp_1, ..., interp_{K-2}, fake] and
        # requires D to assign monotonically decreasing scores along the list.
        # This is purely ADDITIVE to the standard adversarial loss (never replaces it).
        if do_Dmain and self.rank_loss:
            with torch.autograd.profiler.record_function('Drank_forward'):
                # Build ranking list: position 0 = real (best), position K-1 = fake (worst)
                rank_imgs = make_rank_list(
                    real_img, gen_img.detach(),
                    K=self.rank_K, mode=self.rank_mode, alpha_dist=self.rank_alpha_dist
                )
                B_rank, K, C, H, W = rank_imgs.shape
                rank_imgs_flat = rank_imgs.reshape(B_rank * K, C, H, W)

                # Use real_c for scoring (rank list is anchored at real images)
                rank_c = real_c.repeat_interleave(K, dim=0)
                rank_logits = self.run_D(rank_imgs_flat, rank_c, sync=False, augment=self.rank_augment)
                rank_scores = rank_logits.reshape(B_rank, K)

                # Compute ranking loss
                if self.rank_loss_type == 'listmle':
                    loss_Drank = listmle_loss(rank_scores)
                elif self.rank_loss_type == 'pairwise_logistic':
                    loss_Drank = pairwise_logistic_loss(rank_scores)
                elif self.rank_loss_type == 'pairwise_hinge':
                    loss_Drank = pairwise_hinge_loss(rank_scores, margin=self.rank_margin)
                else:
                    loss_Drank = listmle_loss(rank_scores)  # fallback

                training_stats.report('Loss/D/rank', loss_Drank)
            with torch.autograd.profiler.record_function('Drank_backward'):
                (loss_Drank * self.lambda_rank).mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal * self.lambda_adv + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
