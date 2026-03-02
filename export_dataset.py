"""Generate and export synthetic token datasets from pseudocritical HMMs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from experiment_utils import (
    inject_config_defaults,
    resolve_output_dir,
    write_config_snapshot,
)
from hmm_generator import HMMConfig, PseudoCriticalHMM
from stats import (
    estimate_conditional_entropy,
    estimate_pairwise_mi,
    fit_effective_powerlaw,
    plot_conditional_entropy_curve,
    plot_mi_curve,
)


def _save_bin(tokens: np.ndarray, path: Path, vocab_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if vocab_size <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    elif vocab_size <= np.iinfo(np.uint32).max:
        dtype = np.uint32
    else:
        dtype = np.uint64
    arr = tokens.astype(dtype, copy=False)
    mmap = np.memmap(path, mode="w+", dtype=dtype, shape=arr.shape)
    mmap[:] = arr[:]
    mmap.flush()


def _maybe_save_hf_dataset(
    train: np.ndarray, val: np.ndarray, test: np.ndarray, out_dir: Path
) -> bool:
    try:
        from datasets import Dataset, DatasetDict
    except Exception:
        return False

    dsd = DatasetDict(
        {
            "train": Dataset.from_dict({"token": train.tolist()}),
            "validation": Dataset.from_dict({"token": val.tolist()}),
            "test": Dataset.from_dict({"token": test.tolist()}),
        }
    )
    dsd.save_to_disk(str(out_dir / "hf_dataset"))
    return True


def compute_diagnostics(
    tokens: np.ndarray,
    vocab_size: int,
    max_lag: int,
    max_context: int,
    fit_xmin: float,
    fit_xmax: float,
    plot_dir: Path,
) -> dict[str, Any]:
    plot_dir.mkdir(parents=True, exist_ok=True)

    mi = estimate_pairwise_mi(tokens, max_lag=max_lag, vocab_size=vocab_size)
    mi_fit = fit_effective_powerlaw(mi["lags"], mi["mi"], xmin=fit_xmin, xmax=fit_xmax)

    ce = estimate_conditional_entropy(tokens, max_context=max_context, vocab_size=vocab_size)
    H = ce["H"]
    context = ce["context"]
    gap = H[:-1] - H[-1]
    k = context[:-1]
    valid = (k >= 1) & np.isfinite(gap) & (gap > 0)
    ce_fit = fit_effective_powerlaw(k[valid], gap[valid], xmin=1, xmax=max_context - 1)

    plot_mi_curve(mi["lags"], mi["mi"], mi_fit, save_path=plot_dir / "mi_curve.png")
    plot_conditional_entropy_curve(
        ce["context"], ce["H"], ce_fit, save_path=plot_dir / "conditional_entropy_curve.png"
    )

    return {
        "mi": {"lags": mi["lags"].tolist(), "values": mi["mi"].tolist()},
        "conditional_entropy": {
            "context": ce["context"].tolist(),
            "values": ce["H"].tolist(),
        },
        "mi_fit": {
            "success": bool(mi_fit["success"]),
            "slope": float(mi_fit["slope"]),
            "r2": float(mi_fit["r2"]),
        },
        "conditional_entropy_fit": {
            "success": bool(ce_fit["success"]),
            "slope": float(ce_fit["slope"]),
            "r2": float(ce_fit["r2"]),
        },
    }


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_output_dir(
        explicit_out_dir=args.out_dir,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        version=args.version,
        stage="datasets",
        run_name=args.dataset_name,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.hmm_path is not None:
        hmm = PseudoCriticalHMM.load(args.hmm_path)
    else:
        cfg = HMMConfig(
            num_hidden=args.num_hidden,
            vocab_size=args.vocab_size,
            epsilon_min=args.epsilon_min,
            epsilon_max=args.epsilon_max,
            epsilon_schedule=args.epsilon_schedule,
            powerlaw_exponent=args.powerlaw_exponent,
            q_type=args.q_type,
            num_clusters=args.num_clusters,
            cluster_stickiness=args.cluster_stickiness,
            emission_type=args.emission_type,
            eta=args.eta,
            emission_concentration=args.emission_concentration,
            seed=args.seed,
        )
        hmm = PseudoCriticalHMM(cfg)

    tokens = hmm.sample_tokens(args.total_tokens, seed=args.seed)

    n = len(tokens)
    n_train = int(args.train_frac * n)
    n_val = int(args.val_frac * n)
    n_test = n - n_train - n_val

    train = tokens[:n_train]
    val = tokens[n_train : n_train + n_val]
    test = tokens[n_train + n_val :]

    np.save(out_dir / "tokens_all.npy", tokens)
    np.save(out_dir / "train.npy", train)
    np.save(out_dir / "val.npy", val)
    np.save(out_dir / "test.npy", test)

    if args.save_bin:
        _save_bin(tokens, out_dir / "tokens_all.bin", hmm.vocab_size)
        _save_bin(train, out_dir / "train.bin", hmm.vocab_size)
        _save_bin(val, out_dir / "val.bin", hmm.vocab_size)
        _save_bin(test, out_dir / "test.bin", hmm.vocab_size)

    hf_saved = False
    if args.save_hf:
        hf_saved = _maybe_save_hf_dataset(train, val, test, out_dir)

    hmm.save(out_dir / "hmm_model.npz")

    diagnostics = compute_diagnostics(
        tokens=tokens,
        vocab_size=hmm.vocab_size,
        max_lag=args.max_lag,
        max_context=args.max_context,
        fit_xmin=args.fit_xmin,
        fit_xmax=args.fit_xmax,
        plot_dir=out_dir / "plots",
    )

    metadata = {
        "seed": args.seed,
        "num_tokens": int(n),
        "splits": {
            "train": int(n_train),
            "val": int(n_val),
            "test": int(n_test),
        },
        "fractions": {
            "train_frac": float(args.train_frac),
            "val_frac": float(args.val_frac),
            "test_frac": float(1.0 - args.train_frac - args.val_frac),
        },
        "hmm_config": asdict(hmm.config),
        "diagnostics": diagnostics,
        "hf_saved": hf_saved,
        "files": {
            "train": "train.npy",
            "val": "val.npy",
            "test": "test.npy",
            "all": "tokens_all.npy",
        },
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    write_config_snapshot(out_dir, args, getattr(args, "_config_path", None))

    return metadata


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export synthetic HMM token dataset")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=Path("experiments"))
    p.add_argument("--experiment-name", type=str, default="pc_hmm_scaling")
    p.add_argument("--version", type=str, default="v001")
    p.add_argument("--dataset-name", type=str, default="dataset_main")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--hmm-path", type=Path, default=None)

    p.add_argument("--num-hidden", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--epsilon-min", type=float, default=1e-4)
    p.add_argument("--epsilon-max", type=float, default=1e-1)
    p.add_argument("--epsilon-schedule", type=str, default="logspace")
    p.add_argument("--powerlaw-exponent", type=float, default=1.25)
    p.add_argument("--q-type", type=str, default="uniform")
    p.add_argument("--num-clusters", type=int, default=16)
    p.add_argument("--cluster-stickiness", type=float, default=0.92)
    p.add_argument("--emission-type", type=str, default="peaked")
    p.add_argument("--eta", type=float, default=0.15)
    p.add_argument("--emission-concentration", type=float, default=50.0)

    p.add_argument("--total-tokens", type=int, default=2**20)
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--val-frac", type=float, default=0.05)

    p.add_argument("--max-lag", type=int, default=128)
    p.add_argument("--max-context", type=int, default=8)
    p.add_argument("--fit-xmin", type=float, default=2)
    p.add_argument("--fit-xmax", type=float, default=64)

    p.add_argument("--save-bin", action="store_true")
    p.add_argument("--save-hf", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    parser = build_argparser()
    _cfg, cfg_path = inject_config_defaults(parser, sys.argv[1:], section="dataset")
    args = parser.parse_args()
    args._config_path = cfg_path
    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")
    metadata = export_dataset(args)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
