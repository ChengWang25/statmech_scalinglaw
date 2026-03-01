#!/bin/bash
#SBATCH --job-name=setup_llm_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cw5411@princeton.edu

set -euo pipefail

# Della paths
CONDA_BASE="/home/cw5411/miniconda3"
ENV_NAME="LLM_env"
REPO_DIR="/home/cw5411/projects/statmech_scalinglaw"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create env if missing.
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.11
fi

conda activate "${ENV_NAME}"

# Core PyTorch stack for CUDA jobs.
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1

# General transformer-project stack + this project requirements.
pip install --upgrade pip
pip install \
  numpy scipy matplotlib pandas datasets \
  transformers accelerate tokenizers sentencepiece safetensors \
  einops tqdm scikit-learn wandb

pip install -r requirements.txt

python - <<'PY'
import torch, transformers, datasets, numpy, scipy, pandas, matplotlib
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('datasets:', datasets.__version__)
print('environment setup OK')
PY

echo "Environment ${ENV_NAME} is ready."
