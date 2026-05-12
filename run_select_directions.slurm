#!/bin/bash
#SBATCH --job-name=select-directions
#SBATCH --output=logs/select-directions-%j.out
#SBATCH --error=logs/select-directions-%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --constraint=a100|v100
#
# GPU options — override at submission time with sbatch flags:
#   Full A100-80GB (adroit-h11g1):  sbatch --gres=gpu:1            --constraint=gpu80      ...
#   A100 MIG 20GB  (adroit-h11g2):  sbatch --gres=gpu:3g.20gb:1    --constraint=a100       ...
#   V100-32GB      (adroit-h11g3):  sbatch --gres=gpu:1            --constraint=v100       ...
# Note: V100 lacks native bfloat16 compute; PyTorch upcasts to float32 — slower but correct.

# ============================================================
# Validation-based re-selection of RDO/cone candidate pools.
#
# Runs select_rdo_directions.py for one model, scoring every
# vectors_<run_id>.pt pool and writing the best candidate to
# geometry_results_final/results/rdo/<model_name>/selected/.
#
# Usage:
#   sbatch run_select_directions.slurm <model_name> [--run-id <run_id>] [--skip-existing]
#
# <model_name> is the short name used inside geometry_results_final:
#   Qwen2.5-full-lora
#   Qwen2.5-refusal-only-lora
#   Qwen2.5-explanation-only-lora
#   Qwen2.5-justification-only-lora
#
# Examples:
#   sbatch run_select_directions.slurm Qwen2.5-full-lora
#   sbatch run_select_directions.slurm Qwen2.5-refusal-only-lora --skip-existing
#   sbatch run_select_directions.slurm Qwen2.5-full-lora --run-id w0maf5em
# ============================================================

set -euo pipefail

# ── Parse args ───────────────────────────────────────────────
MODEL_NAME="${1:?Usage: sbatch run_select_directions.slurm <model_name> [--run-id <run_id>] [--skip-existing]}"
shift

RUN_ID_ARG=""
SKIP_EXISTING_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)
            RUN_ID_ARG="--run_id ${2:?--run-id requires a value}"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING_ARG="--skip_existing"
            shift
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Model name → full model path mapping ────────────────────
declare -A MODEL_PATH_MAP
MODEL_PATH_MAP["Qwen2.5-full-lora"]="/scratch/network/${USER}/models/Qwen2.5-3B-Instruct-ER-fullweight"
MODEL_PATH_MAP["Qwen2.5-refusal-only-lora"]="/scratch/network/${USER}/models/Qwen2.5-3B-Instruct-ER-fullweight-refusal-only"
MODEL_PATH_MAP["Qwen2.5-explanation-only-lora"]="/scratch/network/${USER}/models/Qwen2.5-3B-Instruct-ER-fullweight-explanation-only"
MODEL_PATH_MAP["Qwen2.5-justification-only-lora"]="/scratch/network/${USER}/models/Qwen2.5-3B-Instruct-ER-fullweight-justification-only"

if [[ -z "${MODEL_PATH_MAP[$MODEL_NAME]+x}" ]]; then
    echo "ERROR: Unknown model_name '${MODEL_NAME}'."
    echo "Valid options: ${!MODEL_PATH_MAP[*]}"
    exit 1
fi

MODEL_PATH="${MODEL_PATH_MAP[$MODEL_NAME]}"

REPO_DIR="/home/${USER}/geometry-of-refusal"
RESULTS_DIR="${REPO_DIR}/geometry_results_final/results"

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Model name:   $MODEL_NAME"
echo "Model path:   $MODEL_PATH"
echo "Results dir:  $RESULTS_DIR"
echo "Run ID filter:${RUN_ID_ARG:-  (all runs)}"
echo "Skip existing:${SKIP_EXISTING_ARG:-  no}"
echo "Start time:   $(date)"
echo "=========================================="

# ── Modules + conda ─────────────────────────────────────────
module purge
module load anaconda3/2024.2
conda activate new-rdo

cd "${REPO_DIR}"

# ── Environment ─────────────────────────────────────────────
export HUGGINGFACE_CACHE_DIR="/scratch/network/${USER}/.cache/huggingface"
export HF_HOME="${HUGGINGFACE_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE_DIR}"
export SAVE_DIR="geometry_results_final/results"
export DIM_DIR="dim_directions"
export WANDB_MODE="offline"
export PYTHONPATH="${REPO_DIR}/refusal_direction:${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p logs

# ── Pre-flight checks ────────────────────────────────────────
echo ""
echo "── Pre-flight checks ── $(date)"

FAIL=0

# 1. Model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "FAIL: model not found at ${MODEL_PATH}"
    FAIL=1
else
    echo "OK:   model at ${MODEL_PATH}"
fi

# 2. Vectors directory exists and is non-empty
VECTORS_DIR="${RESULTS_DIR}/rdo/${MODEL_NAME}/vectors"
if [ ! -d "${VECTORS_DIR}" ]; then
    echo "FAIL: vectors directory not found at ${VECTORS_DIR}"
    FAIL=1
elif [ -z "$(ls "${VECTORS_DIR}"/vectors_*.pt 2>/dev/null)" ]; then
    echo "FAIL: no vectors_*.pt files found in ${VECTORS_DIR}"
    FAIL=1
else
    N_POOLS=$(ls "${VECTORS_DIR}"/vectors_*.pt | wc -l)
    echo "OK:   ${N_POOLS} candidate pool(s) found in ${VECTORS_DIR}"
fi

# 3. DIM direction exists (required to derive add_layer and alpha)
DIM_DIR_PATH="${RESULTS_DIR}/dim_directions/${MODEL_NAME}"
if [ ! -f "${DIM_DIR_PATH}/direction.pt" ]; then
    echo "FAIL: DIM direction.pt not found at ${DIM_DIR_PATH}"
    FAIL=1
else
    echo "OK:   DIM direction.pt at ${DIM_DIR_PATH}"
fi

if [ ! -f "${DIM_DIR_PATH}/direction_metadata.json" ]; then
    echo "FAIL: DIM direction_metadata.json not found at ${DIM_DIR_PATH}"
    FAIL=1
else
    LAYER=$(python3 -c "import json; print(json.load(open('${DIM_DIR_PATH}/direction_metadata.json'))['layer'])")
    echo "OK:   DIM direction_metadata.json — add_layer=${LAYER}"
fi

# 4. Val data exists
for SPLIT in harmful_val harmless_val; do
    VAL_FILE="data/saladbench_splits/${SPLIT}.json"
    if [ ! -f "${VAL_FILE}" ]; then
        echo "FAIL: val data not found at ${VAL_FILE}"
        FAIL=1
    else
        N=$(python3 -c "import json; print(len(json.load(open('${VAL_FILE}'))))")
        echo "OK:   ${VAL_FILE} (${N} examples)"
    fi
done

# 5. select_rdo_directions.py exists
if [ ! -f "${REPO_DIR}/select_rdo_directions.py" ]; then
    echo "FAIL: select_rdo_directions.py not found at ${REPO_DIR}"
    FAIL=1
else
    echo "OK:   select_rdo_directions.py"
fi

# 6. GPU + CUDA
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'OK:   CUDA — {torch.cuda.get_device_name(0)}')" || { echo "FAIL: CUDA unavailable"; FAIL=1; }

echo ""
if [ "${FAIL}" -ne 0 ]; then
    echo "ERROR: one or more pre-flight checks failed. Aborting."
    exit 1
fi
echo "All pre-flight checks passed."

# ── Run selection ────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Running select_rdo_directions.py — $(date)"
echo "=========================================="
echo ""

python select_rdo_directions.py \
    --model_path  "${MODEL_PATH}" \
    --model_name  "${MODEL_NAME}" \
    --results_dir "${RESULTS_DIR}" \
    --kl_threshold 1.0 \
    --batch_size   32 \
    --n_samples    8 \
    ${RUN_ID_ARG} \
    ${SKIP_EXISTING_ARG}

echo ""
echo "=========================================="
echo "DONE for ${MODEL_NAME} at $(date)"
echo "=========================================="
