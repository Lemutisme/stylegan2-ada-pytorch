#!/usr/bin/env bash
#
# Rank Loss Ablation Study for StyleGAN2-ADA
#
# Usage:
#   bash scripts/ablation_rank_loss.sh               # run all ablation groups
#   bash scripts/ablation_rank_loss.sh --dry-run      # validate configs only
#   bash scripts/ablation_rank_loss.sh --group loss    # run a specific group
#
# Requirements:
#   - CUDA-capable GPU(s)
#   - Prepared dataset zip (see dataset_tool.py)
#
# Environment variables (override defaults):
#   DATA        path to dataset zip
#   OUTDIR      output root directory
#   GPUS        number of GPUs
#   CFG         base config preset (auto, cifar, paper256, ...)
#   COND        class-conditional (0 or 1)
#   SNAP        snapshot interval (kimg)
#   KIMG        total training length in kimg
#   METRICS     metrics to evaluate (comma-separated)

set -euo pipefail

# ─────────────────────────────────────────────
# Defaults (override via environment variables)
# ─────────────────────────────────────────────
DATA="${DATA:?ERROR: set DATA=/path/to/dataset.zip}"
OUTDIR="${OUTDIR:-./out/ablation-rank}"
GPUS="${GPUS:-1}"
CFG="${CFG:-cifar}"
COND="${COND:-1}"
SNAP="${SNAP:-10}"
KIMG="${KIMG:-5000}"
METRICS="${METRICS:-fid50k_full}"
DRY_RUN=""
GROUP="all"

# Parse script arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN="--dry-run"; shift ;;
        --group)    GROUP="$2"; shift 2 ;;
        *)          echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
BASE_CMD="python train.py --outdir=${OUTDIR} --data=${DATA} --gpus=${GPUS} \
  --cfg=${CFG} --cond=${COND} --snap=${SNAP} --kimg=${KIMG} --metrics=${METRICS}"

run() {
    local name="$1"; shift
    echo ""
    echo "================================================================"
    echo "  EXPERIMENT: ${name}"
    echo "  $(date)"
    echo "================================================================"
    echo "CMD: ${BASE_CMD} $* ${DRY_RUN}"
    echo ""
    eval ${BASE_CMD} "$@" ${DRY_RUN}
}

should_run() {
    [[ "${GROUP}" == "all" || "${GROUP}" == "$1" ]]
}

# =============================================================
# 0. Baseline: no rank loss (standard StyleGAN2-ADA)
# =============================================================
if should_run "baseline"; then
    run "baseline" \
        --rank-loss=0
fi

# =============================================================
# 1. Loss Type Ablation
#    Fix: K=8, lambda_rank=0.1, mode=intrpl, alpha=linear
# =============================================================
if should_run "loss"; then
    COMMON_LOSS="--rank-loss=1 --rank-k=8 --lambda-rank=0.1 --rank-mode=intrpl --rank-alpha-dist=linear"

    run "loss__listmle" \
        ${COMMON_LOSS} --rank-loss-type=listmle

    run "loss__pairwise_logistic" \
        ${COMMON_LOSS} --rank-loss-type=pairwise_logistic

    run "loss__pairwise_hinge" \
        ${COMMON_LOSS} --rank-loss-type=pairwise_hinge --rank-margin=1.0

    run "loss__lambdaloss" \
        ${COMMON_LOSS} --rank-loss-type=lambdaloss

    run "loss__ppa" \
        ${COMMON_LOSS} --rank-loss-type=ppa --rank-ppa-scale=1.0 --rank-ppa-sinkhorn-iters=0

    run "loss__approxndcg" \
        ${COMMON_LOSS} --rank-loss-type=approxndcg --rank-approxndcg-temperature=1.0
fi

# =============================================================
# 2. Ranking List Length (K) Ablation
#    Fix: listmle, lambda_rank=0.1, mode=intrpl, alpha=linear
# =============================================================
if should_run "K"; then
    COMMON_K="--rank-loss=1 --rank-loss-type=listmle --lambda-rank=0.1 --rank-mode=intrpl --rank-alpha-dist=linear"

    for K in 2 4 8 16; do
        run "K__${K}" \
            ${COMMON_K} --rank-k=${K}
    done
fi

# =============================================================
# 3. Lambda_rank Weight Ablation
#    Fix: listmle, K=8, mode=intrpl, alpha=linear
# =============================================================
if should_run "lambda"; then
    COMMON_LAM="--rank-loss=1 --rank-loss-type=listmle --rank-k=8 --rank-mode=intrpl --rank-alpha-dist=linear"

    for LAM in 0.01 0.05 0.1 0.5 1.0; do
        run "lambda__${LAM}" \
            ${COMMON_LAM} --lambda-rank=${LAM}
    done
fi

# =============================================================
# 4. Interpolation Mode Ablation
#    Fix: listmle, K=8, lambda_rank=0.1, alpha=linear
# =============================================================
if should_run "mode"; then
    COMMON_MODE="--rank-loss=1 --rank-loss-type=listmle --rank-k=8 --lambda-rank=0.1 --rank-alpha-dist=linear"

    run "mode__intrpl" \
        ${COMMON_MODE} --rank-mode=intrpl

    run "mode__noise" \
        ${COMMON_MODE} --rank-mode=noise --rank-noise-scale=0.01

    run "mode__add_mix" \
        ${COMMON_MODE} --rank-mode=add_mix --rank-noise-scale=0.01

    run "mode__seq_mix" \
        ${COMMON_MODE} --rank-mode=seq_mix --rank-noise-scale=0.01
fi

# =============================================================
# 5. Alpha Distribution Ablation
#    Fix: listmle, K=8, lambda_rank=0.1, mode=intrpl
# =============================================================
if should_run "alpha"; then
    COMMON_ALPHA="--rank-loss=1 --rank-loss-type=listmle --rank-k=8 --lambda-rank=0.1 --rank-mode=intrpl"

    for ALPHA in linear cosine random; do
        run "alpha__${ALPHA}" \
            ${COMMON_ALPHA} --rank-alpha-dist=${ALPHA}
    done
fi

# =============================================================
# 6. Adversarial Loss Weight (lambda_adv) Ablation
#    Fix: listmle, K=8, lambda_rank=0.1, mode=intrpl, alpha=linear
#    Tests the balance between rank regularization and adv loss.
# =============================================================
if should_run "adv"; then
    COMMON_ADV="--rank-loss=1 --rank-loss-type=listmle --rank-k=8 --lambda-rank=0.1 --rank-mode=intrpl --rank-alpha-dist=linear"

    for ADV in 0.0 0.5 1.0; do
        run "adv__${ADV}" \
            ${COMMON_ADV} --lambda-adv=${ADV}
    done
fi

# =============================================================
# 7. PPA Hyperparameter Ablation
#    Fix: ppa, K=8, lambda_rank=0.1, mode=intrpl, alpha=linear
# =============================================================
if should_run "ppa"; then
    COMMON_PPA="--rank-loss=1 --rank-loss-type=ppa --rank-k=8 --lambda-rank=0.1 --rank-mode=intrpl --rank-alpha-dist=linear"

    for TAU in 0.1 0.5 1.0 5.0; do
        run "ppa_scale__${TAU}" \
            ${COMMON_PPA} --rank-ppa-scale=${TAU} --rank-ppa-sinkhorn-iters=0
    done

    for SINK in 0 10 50; do
        run "ppa_sinkhorn__${SINK}" \
            ${COMMON_PPA} --rank-ppa-scale=1.0 --rank-ppa-sinkhorn-iters=${SINK}
    done
fi

# =============================================================
# 8. ApproxNDCG Temperature Ablation
#    Fix: approxndcg, K=8, lambda_rank=0.1, mode=intrpl, alpha=linear
# =============================================================
if should_run "approxndcg"; then
    COMMON_ANDCG="--rank-loss=1 --rank-loss-type=approxndcg --rank-k=8 --lambda-rank=0.1 --rank-mode=intrpl --rank-alpha-dist=linear"

    for TEMP in 0.1 0.5 1.0 5.0; do
        run "approxndcg_temp__${TEMP}" \
            ${COMMON_ANDCG} --rank-approxndcg-temperature=${TEMP}
    done
fi

# =============================================================
# 9. Rank Augmentation Ablation
#    Fix: listmle, K=8, lambda_rank=0.1, mode=intrpl, alpha=linear
# =============================================================
if should_run "augment"; then
    COMMON_AUG="--rank-loss=1 --rank-loss-type=listmle --rank-k=8 --lambda-rank=0.1 --rank-mode=intrpl --rank-alpha-dist=linear"

    run "augment__off" \
        ${COMMON_AUG} --rank-augment=0

    run "augment__on" \
        ${COMMON_AUG} --rank-augment=1
fi

echo ""
echo "================================================================"
echo "  All experiments finished.  $(date)"
echo "================================================================"
