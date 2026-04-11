#!/bin/bash
#SBATCH --job-name=eval-rdo
#SBATCH --output=logs/eval-rdo-%j.out
#SBATCH --error=logs/eval-rdo-%j.err
#SBATCH --time=01:00:00
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

export HUGGINGFACE_CACHE_DIR="/scratch/network/${USER}/.cache/huggingface"
export HF_HOME="${HUGGINGFACE_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE_DIR}"

REPO_ROOT="/home/${USER}/geometry-of-refusal"
cd "${REPO_ROOT}/refusal_direction"
export PYTHONPATH="${REPO_ROOT}/refusal_direction:${PYTHONPATH}"

echo ""
echo "Pre-flight checks:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=========================================="
echo "Evaluating RDO direction"
echo "=========================================="
echo ""

python eval_rdo_direction.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --vector_path "${REPO_ROOT}/results/rdo/Qwen2.5-3B-Instruct/vectors/lowest_loss_vector_7ifl0hjw.pt" \
    --dim_metadata_path "${REPO_ROOT}/results/dim_directions/Qwen2.5-3B-Instruct/direction_metadata.json" \
    --output_dir "${REPO_ROOT}/results/rdo/Qwen2.5-3B-Instruct/eval_results"

echo ""
echo "=========================================="
echo "Evaluation complete at $(date)"
echo "=========================================="
