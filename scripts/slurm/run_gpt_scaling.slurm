#!/bin/bash
#SBATCH --job-name=hmm_gpt_scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --constraint=gpu40
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cw5411@princeton.edu

set -euo pipefail

module load cudatoolkit/12.8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate LLM_and_AM

cd /Users/wangcheng/Desktop/statmech_scalinglaw
mkdir -p logs experiments

# Sweep dataset size via array task ID.
DATASET_SIZES=(262144 524288 1048576 2097152 4194304)
NUM_SIZES=${#DATASET_SIZES[@]}
if [[ ${SLURM_ARRAY_TASK_ID} -ge ${NUM_SIZES} ]]; then
  echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} exceeds total jobs ${NUM_SIZES}"
  exit 1
fi

N_TOKENS=${DATASET_SIZES[$SLURM_ARRAY_TASK_ID]}
SEED=0
VOCAB_SIZE=256
BLOCK_SIZE=512
MODEL_PRESET="small"

# Shared dataset (generated once) so array tasks only vary train_tokens_limit.
DATA_ROOT="experiments/slurm_shared_dataset"
LOCK_DIR="${DATA_ROOT}.lock"

if [[ ! -f "${DATA_ROOT}/metadata.json" ]]; then
  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "Generating shared dataset at ${DATA_ROOT}"
    python export_dataset.py \
      --out-dir "${DATA_ROOT}" \
      --num-hidden 512 \
      --vocab-size ${VOCAB_SIZE} \
      --epsilon-min 1e-4 \
      --epsilon-max 1e-1 \
      --epsilon-schedule logspace \
      --q-type cluster \
      --num-clusters 16 \
      --emission-type peaked \
      --eta 0.15 \
      --total-tokens 5000000 \
      --train-frac 0.9 \
      --val-frac 0.05 \
      --max-lag 128 \
      --max-context 8 \
      --seed ${SEED}
    rmdir "${LOCK_DIR}"
  else
    echo "Another task is generating dataset; waiting..."
    until [[ -f "${DATA_ROOT}/metadata.json" ]]; do
      sleep 10
    done
  fi
fi

RUN_ID="N${N_TOKENS}_${MODEL_PRESET}"
OUT_DIR="experiments/slurm_runs/${RUN_ID}"
mkdir -p "${OUT_DIR}"

python train_gpt2.py \
  --train-tokens "${DATA_ROOT}/train.npy" \
  --val-tokens "${DATA_ROOT}/val.npy" \
  --out-dir "${OUT_DIR}" \
  --model-preset "${MODEL_PRESET}" \
  --vocab-size ${VOCAB_SIZE} \
  --block-size ${BLOCK_SIZE} \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --max-iters 3000 \
  --eval-interval 200 \
  --eval-iters 40 \
  --learning-rate 3e-4 \
  --min-lr 3e-5 \
  --warmup-iters 200 \
  --lr-decay-iters 3000 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --train-tokens-limit ${N_TOKENS} \
  --seed ${SEED} \
  --amp

echo "Completed run ${RUN_ID}"
