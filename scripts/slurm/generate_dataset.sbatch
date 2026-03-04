#!/bin/bash
#SBATCH --job-name=gen_hmm_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=sarangg
#SBATCH --array=0-0
#SBATCH --output=/scratch/gpfs/SARANGG/cw5411/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/SARANGG/cw5411/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cw5411@princeton.edu

set -euo pipefail

# ===== User-configurable paths =====
CONDA_BASE="/home/cw5411/miniconda3"
ENV_NAME="LLM_env"
REPO_DIR="/home/cw5411/projects/statmech_scalinglaw"
CONFIG_PATH="${CONFIG_PATH:-configs/experiment_v013.json}"
# Optional manual override; leave empty to use config-driven path.
OUTPUT_ROOT_OVERRIDE="${OUTPUT_ROOT_OVERRIDE:-}"
SCRATCH_ROOT="/scratch/gpfs/SARANGG/cw5411"

# Single-run setting: only the largest dataset size.
DATASET_SIZES=(4194304)
SEQ_LEN="${SEQ_LEN:-8192}"
# Burn-in before recording generated data (default: SEQ_LEN).
BURN_IN=${SEQ_LEN}
SEED="${SEED:-0}"

mkdir -p "${SCRATCH_ROOT}/logs"
cd "${REPO_DIR}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Resolve output root from config to avoid hardcoded version paths.
CFG_OUTPUT_ROOT=$(python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${CONFIG_PATH}").read_text())
print(cfg.get("output_root", "experiments"))
PY
)
CFG_EXPERIMENT_NAME=$(python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${CONFIG_PATH}").read_text())
print(cfg.get("experiment_name", "pc_hmm_scaling"))
PY
)
CFG_VERSION=$(python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${CONFIG_PATH}").read_text())
print(cfg.get("version", "v001"))
PY
)
CFG_MAX_LAG=$(python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${CONFIG_PATH}").read_text())
print(int(cfg.get("dataset", {}).get("max_lag", 8192)))
PY
)

if [[ -n "${OUTPUT_ROOT_OVERRIDE}" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE}"
else
  OUTPUT_ROOT="${CFG_OUTPUT_ROOT}/${CFG_EXPERIMENT_NAME}/${CFG_VERSION}/datasets_grid"
fi

# Add job id namespace to prevent collisions across concurrent submissions.
JOB_TAG=${SLURM_JOB_ID:-manual}
OUTPUT_ROOT="${OUTPUT_ROOT}/job_${JOB_TAG}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_SIZES=${#DATASET_SIZES[@]}
NUM_JOBS=${NUM_SIZES}

if [[ ${TASK_ID} -ge ${NUM_JOBS} ]]; then
  echo "SLURM_ARRAY_TASK_ID ${TASK_ID} exceeds total jobs ${NUM_JOBS}"
  exit 1
fi

TOTAL_TOKENS=${DATASET_SIZES[$TASK_ID]}

RUN_NAME="N${TOTAL_TOKENS}"
OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

CMD=(
  python export_dataset.py
  --config "${CONFIG_PATH}"
  --out-dir "${OUT_DIR}"
  --total-tokens "${TOTAL_TOKENS}"
  --burn-in "${BURN_IN}"
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

summary = {
    "sequence_length": seq_len,
    "burn_in": int(${BURN_IN}),
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

# Stats: use config max_lag (extended window experiments).
MAX_LAG=${CFG_MAX_LAG}
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
