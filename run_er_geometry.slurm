#!/bin/bash
#SBATCH --job-name=er-geometry
#SBATCH --output=logs/er-geometry-%j.out
#SBATCH --error=logs/er-geometry-%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --constraint=gpu80

# ============================================================
# Full geometry pipeline (RDO + Cones + RepInd) for one model.
# Assumes DIM direction already exists in results/dim_directions/.
#
# Usage:
#   sbatch run_er_geometry.slurm /scratch/network/$USER/models/Qwen2.5-3B-Instruct-ER-fullweight
#   sbatch run_er_geometry.slurm /scratch/network/$USER/models/Qwen2.5-3B-Instruct-ER-fullweight-refusal-only
#   sbatch run_er_geometry.slurm /scratch/network/$USER/models/Qwen2.5-3B-Instruct-ER-fullweight-explanation-only
#   sbatch run_er_geometry.slurm /scratch/network/$USER/models/Qwen2.5-3B-Instruct-ER-fullweight-justification-only
#
# ============================================================

MODEL_PATH="${1:?Usage: sbatch run_er_geometry.slurm <model_path>}"
MODEL_NAME=$(basename "${MODEL_PATH}")

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Model:        $MODEL_NAME"
echo "Model path:   $MODEL_PATH"
echo "Start time:   $(date)"
echo "=========================================="

module purge
module load anaconda3/2024.2

conda activate rdo

export HUGGINGFACE_CACHE_DIR="/scratch/network/${USER}/.cache/huggingface"
export HF_HOME="${HUGGINGFACE_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE_DIR}"

export SAVE_DIR="results"
export DIM_DIR="dim_directions"
export WANDB_MODE="offline"

cd /home/${USER}/geometry-of-refusal

echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Verify DIM direction exists
DIM_PATH="${SAVE_DIR}/${DIM_DIR}/${MODEL_NAME}"
if [ ! -f "${DIM_PATH}/direction.pt" ] || [ ! -f "${DIM_PATH}/direction_metadata.json" ]; then
    echo "ERROR: DIM direction not found at ${DIM_PATH}"
    echo "Run the DIM pipeline first, or bootstrap from another model."
    exit 1
fi
echo "DIM direction: OK (${DIM_PATH})"

# ── 1. RDO (single direction) ───────────────────────────────
echo ""
echo "=========================================="
echo "[1/3] RDO direction — $(date)"
echo "=========================================="
echo ""

python rdo.py \
    --model "${MODEL_PATH}" \
    --train_direction \
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

# ── 2. Refusal cones (dim 2-6) ──────────────────────────────
echo ""
echo "=========================================="
echo "[2/3] Cone training (dim 2-6) — $(date)"
echo "=========================================="
echo ""

python rdo.py \
    --model "${MODEL_PATH}" \
    --train_cone \
    --min_cone_dim 2 \
    --max_cone_dim 6 \
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

# ── 3. RepInd directions ────────────────────────────────────
echo ""
echo "=========================================="
echo "[3/3] RepInd training — $(date)"
echo "=========================================="
echo ""

python rdo_repind.py \
    --model "${MODEL_PATH}" \
    --train_independent_direction \
    --epochs 2 \
    --lr 0.01 \
    --batch_size 1 \
    --effective_batch_size 16 \
    --patience 5 \
    --n_lr_reduce 2 \
    || echo "WARNING: RepInd failed"

echo ""
echo "=========================================="
echo "ALL COMPLETE for ${MODEL_NAME} at $(date)"
echo "=========================================="
