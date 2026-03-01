# Pseudocritical HMM -> GPT Scaling Law Project

Reproducible Python project for:
- generating synthetic token sequences from a large finite-state pseudocritical HMM,
- measuring information-theoretic diagnostics,
- training GPT-2-like causal language models from scratch,
- fitting loss-vs-dataset-size scaling laws.

## Why this HMM is "pseudocritical"

The hidden transition is parameterized as:
- `P[i,i] = 1 - epsilon_i`
- `P[i,j] = epsilon_i * Q[i,j]` for `j != i`

Using many hidden states and a dense spread of `epsilon_i` (e.g. log-spaced between `1e-4` and `1e-1`) creates a broad timescale mixture. Over finite lags this can mimic slow, critical-like decay in observables without requiring an actual infinite critical system.

## File Tree

```text
.
├── README.md
├── requirements.txt
├── hmm_generator.py
├── stats.py
├── export_dataset.py
├── train_gpt2.py
├── sweep.py
└── analyze_scaling.py
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Core Modules

- `hmm_generator.py`
  - `PseudoCriticalHMM` and `HMMConfig`
  - epsilon schedules: `logspace`, `powerlaw`, `custom`
  - Q mixing: `uniform`, `cluster`, `random`
  - emissions: `onehot`, `peaked`, `dirichlet_cluster`
  - methods: `sample_hidden`, `sample_observed`, `sample_tokens`, `stationary_distribution`, `save/load`

- `stats.py`
  - `estimate_pairwise_mi`
  - `estimate_conditional_entropy` (exact n-gram)
  - `estimate_autocorrelation`
  - `estimate_transition_spectrum`
  - `fit_effective_powerlaw`
  - plotting helpers for MI, entropy, spectrum

- `export_dataset.py`
  - generates long token sequence
  - contiguous `train/val/test` split
  - exports `.npy`, optional `.bin`, optional Hugging Face datasets
  - stores metadata JSON with generator config + measured MI/entropy fit slopes

- `train_gpt2.py`
  - GPT-2-like causal transformer from scratch
  - configurable `n_layer`, `n_head`, `n_embd`, `block_size`, `dropout`, `bias`
  - mixed precision (CUDA), grad accumulation, AdamW, cosine LR decay + warmup,
    grad clipping, periodic evaluation, checkpoints

- `sweep.py`
  - resumable sweeps over HMM params, dataset sizes, model sizes
  - records run metadata and losses to structured directories + aggregate JSONL

- `analyze_scaling.py`
  - fits `L(N) = L_inf + A * N^{-alpha}`
  - reports parameter CI (curve-fit covariance approximation)
  - plots loss curves and `alpha` correlations vs MI/entropy slopes

## Suggested Defaults

HMM defaults:
- `num_hidden=512`
- `vocab_size=256`
- `epsilon_min=1e-4`
- `epsilon_max=1e-1`
- `epsilon_schedule=logspace`
- `eta=0.15`

GPT presets:
- `small`: 6 layers, 4 heads, 256 embd
- `medium`: 8 layers, 6 heads, 384 embd
- `large`: 12 layers, 8 heads, 512 embd

## Minimal End-to-End Example (single-GPU friendly)

### 1) Create and save a pseudocritical HMM

```bash
python hmm_generator.py \
  --out artifacts/hmm_model.npz \
  --num-hidden 512 \
  --vocab-size 256 \
  --epsilon-min 1e-4 \
  --epsilon-max 1e-1 \
  --epsilon-schedule logspace \
  --q-type cluster \
  --num-clusters 16 \
  --emission-type peaked \
  --eta 0.15 \
  --seed 0
```

### 2) Generate dataset + diagnostics

```bash
python export_dataset.py \
  --out-dir experiments/demo_dataset \
  --hmm-path artifacts/hmm_model.npz \
  --total-tokens 262144 \
  --train-frac 0.9 \
  --val-frac 0.05 \
  --max-lag 64 \
  --max-context 6 \
  --seed 0
```

### 3) Measure MI and conditional entropy directly

```bash
python stats.py \
  --tokens experiments/demo_dataset/train.npy \
  --vocab-size 256 \
  --max-lag 64 \
  --max-context 6 \
  --out-dir experiments/demo_stats
```

### 4) Train small GPT from scratch

```bash
python train_gpt2.py \
  --train-tokens experiments/demo_dataset/train.npy \
  --val-tokens experiments/demo_dataset/val.npy \
  --out-dir experiments/demo_train_small \
  --model-preset small \
  --block-size 256 \
  --batch-size 16 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 20 \
  --learning-rate 3e-4 \
  --warmup-iters 100 \
  --lr-decay-iters 1000 \
  --amp
```

### 5) Run a compact scaling sweep and fit scaling law

```bash
python sweep.py \
  --out-root experiments/demo_sweep \
  --num-hidden-grid 512 \
  --epsilon-min-grid 1e-4 \
  --epsilon-max-grid 1e-1 \
  --epsilon-schedule-grid logspace \
  --eta-grid 0.15 \
  --vocab-size-grid 256 \
  --dataset-sizes 65536,131072,262144 \
  --model-sizes small \
  --total-tokens 300000 \
  --block-size 256 \
  --max-iters 600 \
  --eval-interval 100 \
  --eval-iters 20

python analyze_scaling.py \
  --experiment-root experiments/demo_sweep \
  --out-dir experiments/demo_analysis \
  --min-points 3
```

## Notes

- Diagnostics are empirical and finite-sample; for publication-quality numbers, use longer sequences.
- Exact n-gram conditional entropy scales poorly with large `max_context`; defaults keep it tractable.
- Sweep is resumable: existing run directories with `result.json` are skipped.
- This project trains from scratch and does not use pretrained Hugging Face weights.
