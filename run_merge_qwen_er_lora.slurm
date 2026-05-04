#!/bin/bash
#SBATCH --job-name=merge-qwen-er-lora
#SBATCH --output=logs/merge-qwen-er-lora-%j.out
#SBATCH --error=logs/merge-qwen-er-lora-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
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

MERGE_USER="${MERGE_USER:-$USER}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
HUGGINGFACE_CACHE_DIR="${HUGGINGFACE_CACHE_DIR:-/scratch/network/${MERGE_USER}/.cache/huggingface}"
MODELS_DIR="${MODELS_DIR:-/scratch/network/${MERGE_USER}/models}"

export HUGGINGFACE_CACHE_DIR
export HF_HOME="${HUGGINGFACE_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE_DIR}"

REPO_ROOT="/home/${USER}/geometry-of-refusal"
cd "${REPO_ROOT}"

mkdir -p logs
mkdir -p "${HUGGINGFACE_CACHE_DIR}"
mkdir -p "${MODELS_DIR}"

echo ""
echo "Pre-flight checks:"
echo "  MERGE_USER: ${MERGE_USER}"
echo "  BASE_MODEL: ${BASE_MODEL}"
echo "  HF cache:   ${HUGGINGFACE_CACHE_DIR}"
echo "  Models dir: ${MODELS_DIR}"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers, peft; print('transformers and peft OK')"

echo ""
echo "=========================================="
echo "Starting LoRA merge"
echo "=========================================="
echo ""

python merge_qwen_er_lora.py \
    --user "${MERGE_USER}" \
    --base-model "${BASE_MODEL}" \
    --cache-dir "${HUGGINGFACE_CACHE_DIR}" \
    --models-dir "${MODELS_DIR}" \
    --device-map cpu

echo ""
echo "=========================================="
echo "Merge complete at $(date)"
echo "=========================================="
