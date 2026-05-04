#!/bin/bash
#SBATCH --job-name=dim-qwen2.5-3b-er
#SBATCH --output=logs/dim-qwen2.5-3b-er-%j.out
#SBATCH --error=logs/dim-qwen2.5-3b-er-%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --constraint=a100

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPUs:         $SLURM_GPUS_ON_NODE"
echo "Start time:   $(date)"
echo "=========================================="

module purge
module load anaconda3/2024.2

conda activate qwen-er

export HF_HUB_OFFLINE=1
export HUGGINGFACE_CACHE_DIR="/scratch/network/${USER}/.cache/huggingface"
export HF_HOME="${HUGGINGFACE_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE_DIR}"
mkdir -p "${HUGGINGFACE_CACHE_DIR}"

export SAVE_DIR="results"
export DIM_DIR="dim_directions"

REPO_ROOT="/home/${USER}/geometry-of-refusal"

cd "${REPO_ROOT}/refusal_direction"
export PYTHONPATH="${REPO_ROOT}/refusal_direction:${PYTHONPATH}"

mkdir -p "${REPO_ROOT}/logs"

echo ""
echo "Pre-flight checks:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
for f in "${REPO_ROOT}/data/saladbench_splits/harmful_train.json" \
         "${REPO_ROOT}/data/saladbench_splits/harmless_train.json" \
         "${REPO_ROOT}/refusal_direction/dataset/processed/jailbreakbench.json"; do
    if [ -f "$f" ]; then
        echo "  OK: $(basename $f)"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "=========================================="
echo "Starting DIM pipeline"
echo "=========================================="
echo ""

python pipeline/run_pipeline.py --model_path /scratch/network/mp3687/models/Qwen2.5-3B-Instruct-ER-fullweight --no_filter

echo ""
echo "=========================================="
echo "Pipeline complete at $(date)"
echo "=========================================="

EVAL_DIR="${REPO_ROOT}/${SAVE_DIR}/${DIM_DIR}/Qwen/Qwen2.5-3B-Instruct-ER-fullweight/completions"
if [ -d "$EVAL_DIR" ]; then
    echo ""
    echo "Results:"
    for f in "$EVAL_DIR"/*_evaluations.json; do
        if [ -f "$f" ]; then
            echo "  $(basename $f):"
            python -c "
import json
with open('$f') as fh:
    d = json.load(fh)
for k, v in d.items():
    if k == 'completions' or 'per_category' in k:
        continue
    print(f'    {k}: {v}')
"
        fi
    done
fi
