#!/bin/bash
#SBATCH --job-name=gen_hmm_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
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
OUTPUT_ROOT="experiments/pc_hmm_scaling/v001/datasets_grid"

# Pseudocritical lag window threshold used for labeling sequence regimes.
PSEUDO_CRITICAL_WINDOW=4096

# Grid: dataset total token budget and fixed sequence length.
# Keep array range in sync with total combinations.
DATASET_SIZES=(262144 524288 1048576 2097152)
SEQ_LENGTHS=(1024 4096 8192)
SEED=0

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_SIZES=${#DATASET_SIZES[@]}
NUM_SEQS=${#SEQ_LENGTHS[@]}
NUM_JOBS=$((NUM_SIZES * NUM_SEQS))

if [[ ${TASK_ID} -ge ${NUM_JOBS} ]]; then
  echo "SLURM_ARRAY_TASK_ID ${TASK_ID} exceeds total jobs ${NUM_JOBS}"
  exit 1
fi

SIZE_IDX=$((TASK_ID / NUM_SEQS))
SEQ_IDX=$((TASK_ID % NUM_SEQS))
TOTAL_TOKENS=${DATASET_SIZES[$SIZE_IDX]}
SEQ_LEN=${SEQ_LENGTHS[$SEQ_IDX]}

if [[ ${SEQ_LEN} -lt ${PSEUDO_CRITICAL_WINDOW} ]]; then
  REGIME_TAG="subwindow"
elif [[ ${SEQ_LEN} -eq ${PSEUDO_CRITICAL_WINDOW} ]]; then
  REGIME_TAG="atwindow"
else
  REGIME_TAG="superwindow"
fi

RUN_NAME="N${TOTAL_TOKENS}_L${SEQ_LEN}_${REGIME_TAG}"
OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

CMD=(
  python export_dataset.py
  --config "${CONFIG_PATH}"
  --out-dir "${OUT_DIR}"
  --total-tokens "${TOTAL_TOKENS}"
  --dataset-name "${RUN_NAME}"
  --seed "${SEED}"
)

echo "Task ${TASK_ID}/${NUM_JOBS}: ${RUN_NAME}"
echo "Running export: ${CMD[*]}"
"${CMD[@]}"

# Create fixed-length sequence views for each split.
python - <<PY
import json
from pathlib import Path
import numpy as np

out_dir = Path("${OUT_DIR}")
seq_len = int(${SEQ_LEN})
pc_window = int(${PSEUDO_CRITICAL_WINDOW})

summary = {
    "sequence_length": seq_len,
    "pseudo_critical_window": pc_window,
    "regime": "${REGIME_TAG}",
    "splits": {},
}

for split in ("train", "val", "test"):
    arr = np.load(out_dir / f"{split}.npy")
    n_seq = len(arr) // seq_len
    kept = n_seq * seq_len
    dropped = len(arr) - kept
    seq = arr[:kept].reshape(n_seq, seq_len) if n_seq > 0 else np.empty((0, seq_len), dtype=arr.dtype)
    np.save(out_dir / f"{split}_seq_len{seq_len}.npy", seq)
    summary["splits"][split] = {
        "num_tokens_raw": int(len(arr)),
        "num_tokens_kept": int(kept),
        "num_tokens_dropped": int(dropped),
        "num_sequences": int(n_seq),
        "seq_file": f"{split}_seq_len{seq_len}.npy",
    }

with (out_dir / "sequence_view_metadata.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
PY

# Optional stats pass tuned to sequence-length regime.
MAX_LAG=${SEQ_LEN}
if [[ ${MAX_LAG} -gt 16384 ]]; then
  MAX_LAG=16384
fi
VOCAB_SIZE=$(python - <<PY
import json
from pathlib import Path
meta = Path("${OUT_DIR}") / "metadata.json"
with meta.open("r", encoding="utf-8") as f:
    obj = json.load(f)
print(int(obj["hmm_config"]["vocab_size"]))
PY
)

python stats.py \
  --tokens "${OUT_DIR}/train.npy" \
  --vocab-size "${VOCAB_SIZE}" \
  --max-lag "${MAX_LAG}" \
  --max-context 8 \
  --out-dir "${OUT_DIR}/stats"

echo "Dataset generation complete for ${RUN_NAME}."
