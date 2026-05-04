#!/bin/bash
# ============================================================
# Download ER model variants to scratch
#
# Usage:
#   conda activate new-rdo
#   python download_er_models.py
#
# ============================================================

#SBATCH --job-name=download-er
#SBATCH --output=logs/download-er-%j.out
#SBATCH --error=logs/download-er-%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

module purge
module load anaconda3/2024.2

conda activate new-rdo

cd /home/${USER}/geometry-of-refusal

python download_er_models.py
