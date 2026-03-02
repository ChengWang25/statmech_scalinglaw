#!/bin/bash
#SBATCH --job-name=hmm_gpt_scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=gpu40
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cw5411@princeton.edu

set -euo pipefail

CONDA_BASE="/home/cw5411/miniconda3"
ENV_NAME="LLM_env"
REPO_DIR="/home/cw5411/statmech_scalinglaw"
CONFIG_PATH="configs/experiment_v001.json"

mkdir -p "${REPO_DIR}/logs"

module purge
module load cudatoolkit/12.8

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${REPO_DIR}"

python sweep.py --config "${CONFIG_PATH}"
python analyze_scaling.py --config "${CONFIG_PATH}"

echo "Sweep + analysis completed with config ${CONFIG_PATH}"
