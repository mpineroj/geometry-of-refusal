#!/bin/bash
#SBATCH --job-name=dim-qwen3b
#SBATCH --output=logs/dim-qwen3b-%j.out
#SBATCH --error=logs/dim-qwen3b-%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

###############################################################################
# DIM Pipeline -- Qwen2.5-3B-Instruct on Princeton Adroit
#
# Usage:
#   mkdir -p logs
#   sbatch run_dim_della.slurm
#
# BEFORE FIRST RUN -- check/adjust these:
#   1. Partition: uncomment the right line below for your allocation
#   2. GPU type: A100 preferred, V100-32GB also works for DIM
#   3. Module loads: check `module avail` for exact names
#   4. Conda env: adjust the env name or use venv instead
#   5. HuggingFace cache: point to scratch so weights aren't in /home
###############################################################################

# --- Partition (uncomment ONE, check with: sshare -u $USER) ---
# SBATCH --partition=gpu
# #SBATCH --partition=mig
# SBATCH --constraint=a100

###############################################################################
# Environment setup
###############################################################################

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPUs:         $SLURM_GPUS_ON_NODE"
echo "Start time:   $(date)"
echo "Working dir:  $(pwd)"
echo "=========================================="

# Load modules -- adjust versions with `module avail anaconda` / `module avail cuda`
module purge
module load anaconda3/2024.2
# module load cudatoolkit/12.2    # often auto-loaded with anaconda

# Activate conda env
# First-time setup:
#   conda create -n refusal python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
#   conda activate refusal
#   pip install transformers accelerate datasets python-dotenv jaxtyping einops wandb
#   pip install strong-reject    # optional, only for StrongREJECT eval
conda activate rdo        

# HuggingFace cache -> scratch
export HUGGINGFACE_CACHE_DIR="/scratch/network/${USER}/.cache/huggingface"
export HF_HOME="${HUGGINGFACE_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE_DIR}"
mkdir -p "${HUGGINGFACE_CACHE_DIR}"

# Pipeline output directories
export SAVE_DIR="results"
export DIM_DIR="dim_directions"

# Repo directory
cd /home/${USER}/geometry-of-refusal

mkdir -p "${SAVE_DIR}/${DIM_DIR}"
mkdir -p logs

###############################################################################
# Pre-flight checks
###############################################################################

echo ""
echo "Pre-flight checks:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"

echo ""
for f in data/saladbench_splits/harmful_train.json data/saladbench_splits/harmless_train.json dataset/processed/jailbreakbench.json; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f -- pipeline will fail"
    fi
done

###############################################################################
# Run
###############################################################################

echo ""
echo "=========================================="
echo "Starting DIM pipeline"
echo "=========================================="
echo ""

python pipeline/run_pipeline.py --model_path Qwen/Qwen2.5-3B-Instruct

echo ""
echo "=========================================="
echo "Pipeline complete at $(date)"
echo "=========================================="

# Print results summary
EVAL_DIR="${SAVE_DIR}/${DIM_DIR}/Qwen2.5-3B-Instruct/completions"
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
