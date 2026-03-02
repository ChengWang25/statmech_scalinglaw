#!/bin/bash
#SBATCH --job-name=gen_hmm_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cw5411@princeton.edu

set -euo pipefail

# ===== User-configurable paths =====
CONDA_BASE="/home/cw5411/miniconda3"
ENV_NAME="LLM_env"
REPO_DIR="/home/cw5411/statmech_scalinglaw"
CONFIG_PATH="configs/experiment_v001.json"

# Optional direct overrides (keep empty to use config values)
OUT_DIR=""
TOTAL_TOKENS=""        # e.g. 1048576
DATASET_NAME=""        # e.g. dataset_main
SEED=""                # e.g. 0

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

CMD=(python export_dataset.py --config "${CONFIG_PATH}")

if [[ -n "${OUT_DIR}" ]]; then
  CMD+=(--out-dir "${OUT_DIR}")
fi
if [[ -n "${TOTAL_TOKENS}" ]]; then
  CMD+=(--total-tokens "${TOTAL_TOKENS}")
fi
if [[ -n "${DATASET_NAME}" ]]; then
  CMD+=(--dataset-name "${DATASET_NAME}")
fi
if [[ -n "${SEED}" ]]; then
  CMD+=(--seed "${SEED}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Dataset generation complete."
