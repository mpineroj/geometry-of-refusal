#!/bin/bash
#SBATCH --job-name=rdo-qwen3b
#SBATCH --output=logs/rdo-qwen3b-%j.out
#SBATCH --error=logs/rdo-qwen3b-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
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

export HUGGINGFACE_CACHE_DIR="/scratch/network/${USER}/.cache/huggingface"
export HF_HOME="${HUGGINGFACE_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE_DIR}"

export SAVE_DIR="results"
export DIM_DIR="dim_directions"
export WANDB_MODE="offline"

cd /home/${USER}/geometry-of-refusal

echo ""
echo "Pre-flight checks:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "=========================================="
echo "Starting RDO training (single direction)"
echo "=========================================="
echo ""

python rdo.py \
    --model Qwen/Qwen2.5-3B-Instruct \
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
    --filter_data

echo ""
echo "=========================================="
echo "RDO training complete at $(date)"
echo "=========================================="