#!/bin/bash
#SBATCH --job-name=rdo-test
#SBATCH --output=logs/rdo-test-%j.out
#SBATCH --error=logs/rdo-test-%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --constraint=gpu80

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Start time:   $(date)"
echo "=========================================="

module purge
module load anaconda3/2024.2

conda activate rdo

# HF_HUB_OFFLINE disabled — nnsight needs cache lookup without strict offline mode
export HUGGINGFACE_CACHE_DIR="/scratch/network/${USER}/.cache/huggingface"
export HF_HOME="${HUGGINGFACE_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE_DIR}"

# Load .env for WANDB_PROJECT, WANDB_ENTITY, SAVE_DIR, DIM_DIR
REPO_ROOT="/home/${USER}/geometry-of-refusal"
cd "${REPO_ROOT}"

export SAVE_DIR="results"
export DIM_DIR="dim_directions"
export WANDB_MODE=offline
# Select model at submit-time, e.g.:
#   MODEL="Qwen/Qwen2.5-3B-Instruct" sbatch run_rdo_test.slurm
#   MODEL="rpawar7156/qwen2.5-3b-er" sbatch run_rdo_test.slurm
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_ID="${MODEL##*/}"
MODEL_CACHE_DIR="${HUGGINGFACE_CACHE_DIR}/models--${MODEL//\//--}"

export MODEL


echo ""
echo "Pre-flight checks:"
echo "Model:        ${MODEL}"
echo "Model ID:     ${MODEL_ID}"
if [ -d "${MODEL_CACHE_DIR}" ]; then
    echo "Cache path:   ${MODEL_CACHE_DIR}"
else
    echo "Cache path:   MISSING (${MODEL_CACHE_DIR})"
    echo "Available cached models under ${HUGGINGFACE_CACHE_DIR}:"
    ls -1 "${HUGGINGFACE_CACHE_DIR}" | grep '^models--' || true
    echo "Aborting: selected MODEL is not present in cache."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "import nnsight; print(f'nnsight OK')"
python -c "import wandb; print(f'wandb OK')"

echo ""
echo "Checking DIM artifacts:"
for f in "${SAVE_DIR}/${DIM_DIR}/${MODEL}/direction.pt" \
         "${SAVE_DIR}/${DIM_DIR}/${MODEL}/direction_metadata.json" \
         "${SAVE_DIR}/${DIM_DIR}/${MODEL}/generate_directions/mean_diffs.pt"; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f -- RDO will fail"
    fi
done

echo ""
echo "Checking data files:"
for f in "data/saladbench_splits/harmful_train.json" \
         "data/saladbench_splits/harmless_train.json"; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f -- RDO will fail"
    fi
done

echo ""
echo "=========================================="
echo "Starting RDO test (short run)"
echo "=========================================="
echo ""

python rdo.py \
    --model "${MODEL}" \
    --train_direction \
    --epochs 1 \
    --effective_batch_size 2 \
    --patience 1 \
    --n_lr_reduce 0 \
    --filter_data

echo ""
echo "=========================================="
echo "RDO test complete at $(date)"
echo "=========================================="