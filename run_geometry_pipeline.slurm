#!/bin/bash
#SBATCH --job-name=geometry-pipeline
#SBATCH --output=logs/geometry-pipeline-%j.out
#SBATCH --error=logs/geometry-pipeline-%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --constraint=gpu80

# ============================================================
# Full geometry pipeline (DIM + RDO + Cones + RepInd) for one model.
#
#
# Usage:
#   sbatch run_geometry_pipeline.slurm <model_path> \
#       [--skip-dim] [--skip-rdo] [--skip-cones] [--skip-repind]
#
# Examples:
#   sbatch run_geometry_pipeline.slurm \
#       /scratch/network/$USER/models/Qwen2.5-3B-Instruct-ER-fullweight
#
#   sbatch run_geometry_pipeline.slurm \
#       /scratch/network/$USER/models/Qwen2.5-3B-Instruct-ER-fullweight \
#       --skip-dim --skip-repind
# ============================================================

set -u

# ── Parse args ──────────────────────────────────────────────
MODEL_PATH="${1:?Usage: sbatch run_geometry_pipeline.slurm <model_path> [--skip-dim] [--skip-rdo] [--skip-cones] [--skip-repind]}"
shift

SKIP_DIM=0
SKIP_RDO=0
SKIP_CONES=0
SKIP_REPIND=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-dim)    SKIP_DIM=1;    shift ;;
        --skip-rdo)    SKIP_RDO=1;    shift ;;
        --skip-cones)  SKIP_CONES=1;  shift ;;
        --skip-repind) SKIP_REPIND=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODEL_NAME=$(basename "${MODEL_PATH}")

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Model:        $MODEL_NAME"
echo "Model path:   $MODEL_PATH"
echo "Skip DIM:     $SKIP_DIM"
echo "Skip RDO:     $SKIP_RDO"
echo "Skip Cones:   $SKIP_CONES"
echo "Skip RepInd:  $SKIP_REPIND"
echo "Start time:   $(date)"
echo "=========================================="

# ── Modules + conda ─────────────────────────────────────────
module purge
module load anaconda3/2024.2
conda activate new-rdo

REPO_DIR="/home/${USER}/geometry-of-refusal"
cd "${REPO_DIR}"

# ── Environment ─────────────────────────────────────────────
export HUGGINGFACE_CACHE_DIR="/scratch/network/${USER}/.cache/huggingface"
export HF_HOME="${HUGGINGFACE_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE_DIR}"

export SAVE_DIR="geometry_results_final/results"
export DIM_DIR="dim_directions"

# Modal uses online wandb; flip to "offline" if Adroit compute nodes
# can't reach the wandb API on your current setup.
export WANDB_MODE="offline"
export WANDB_ENTITY="mpinero-princeton-university"
export WANDB_PROJECT="refusal_directions"
export WANDB_DIR="${SAVE_DIR}/wandb"

export PYTHONPATH="${REPO_DIR}/refusal_direction:${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p "${WANDB_DIR}" "logs"

# Write .env so rdo.py's dotenv.load_dotenv picks up the same values
# the Modal script injects via its env dict.
cat > "${REPO_DIR}/.env" <<EOF
HUGGINGFACE_CACHE_DIR="${HUGGINGFACE_CACHE_DIR}"
SAVE_DIR="${SAVE_DIR}"
DIM_DIR="${DIM_DIR}"
WANDB_ENTITY="${WANDB_ENTITY}"
WANDB_PROJECT="${WANDB_PROJECT}"
WANDB_MODE=${WANDB_MODE}
EOF

echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# ── Pre-flight checks ───────────────────────────────────────
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    exit 1
fi
echo "Model: OK (${MODEL_PATH})"

DIM_PATH="${SAVE_DIR}/${DIM_DIR}/${MODEL_NAME}"

# ── 0. DIM direction extraction ─────────────────────────────
if [ "${SKIP_DIM}" -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[0/3] DIM direction extraction — $(date)"
    echo "=========================================="
    echo ""

    if [ -f "${DIM_PATH}/direction.pt" ] && [ -f "${DIM_PATH}/direction_metadata.json" ]; then
        echo "DIM direction already exists for ${MODEL_NAME}, overwriting with new one."
    else
        echo "No existing DIM direction found for ${MODEL_NAME}, extracting new one."
    fi

    (
        cd "${REPO_DIR}/refusal_direction"
        python pipeline/run_pipeline.py \
            --model_path "${MODEL_PATH}" \
            --no_filter \
            || echo "WARNING: DIM extraction failed, continuing..."
    )

    echo ""
    echo "── DIM complete ── $(date)"
else
    echo "Skipping [0/3] DIM direction extraction"
fi

# DIM artifacts are required for any downstream stage
if [ "${SKIP_RDO}" -eq 0 ] || [ "${SKIP_CONES}" -eq 0 ] || [ "${SKIP_REPIND}" -eq 0 ]; then
    if [ ! -f "${DIM_PATH}/direction.pt" ] || [ ! -f "${DIM_PATH}/direction_metadata.json" ]; then
        echo "ERROR: DIM direction not found at ${DIM_PATH}"
        echo "Re-run with DIM enabled, or bootstrap from another model."
        exit 1
    fi
    echo "DIM direction: OK (${DIM_PATH})"
fi

# ── 1. RDO (single direction) ───────────────────────────────
if [ "${SKIP_RDO}" -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[1/3] RDO direction — $(date)"
    echo "=========================================="
    echo ""

    python rdo.py \
        --model "${MODEL_PATH}" \
        --train_direction \
        --target_generation_batch_size 32 \
        --epochs 1 \
        --lr 0.01 \
        --batch_size 1 \
        --effective_batch_size 16 \
        --patience 5 \
        --n_lr_reduce 2 \
        --ablation_lambda 1.0 \
        --addition_lambda 0.2 \
        --retain_lambda 1.0 \
        || echo "WARNING: RDO failed, continuing..."

    echo ""
    echo "── RDO complete ── $(date)"
else
    echo "Skipping [1/3] RDO direction"
fi

# ── 2. Refusal cones ──────────────────────────────
if [ "${SKIP_CONES}" -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[2/3] Cone training (2-5) — $(date)"
    echo "=========================================="
    echo ""

    python rdo.py \
        --model "${MODEL_PATH}" \
        --train_cone \
        --target_generation_batch_size 32 \
        --min_cone_dim 2 \
        --max_cone_dim 5 \
        --epochs 1 \
        --lr 0.01 \
        --batch_size 1 \
        --effective_batch_size 16 \
        --patience 5 \
        --n_lr_reduce 2 \
        --ablation_lambda 1.0 \
        --addition_lambda 0.2 \
        --retain_lambda 1.0 \
        --n_sample 8 \
        --fixed_samples 8 \
        || echo "WARNING: Cone training failed, continuing..."

    echo ""
    echo "── Cones complete ── $(date)"
else
    echo "Skipping [2/3] Cone training"
fi

# ── 3. RepInd directions ────────────────────────────────────
if [ "${SKIP_REPIND}" -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[3/3] RepInd training — $(date)"
    echo "=========================================="
    echo ""

    python rdo_repind.py \
        --model "${MODEL_PATH}" \
        --train_independent_direction \
        --target_generation_batch_size 32 \
        --epochs 2 \
        --lr 0.01 \
        --batch_size 1 \
        --effective_batch_size 16 \
        --patience 5 \
        --n_lr_reduce 2 \
        || echo "WARNING: RepInd failed"

    echo ""
    echo "── RepInd complete ── $(date)"
else
    echo "Skipping [3/3] RepInd training"
fi

echo ""
echo "=========================================="
echo "ALL COMPLETE for ${MODEL_NAME} at $(date)"
echo "=========================================="