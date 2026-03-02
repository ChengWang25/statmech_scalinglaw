"""Resumable experiment sweeps over HMM parameters, dataset sizes, and GPT sizes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiment_utils import (
    inject_config_defaults,
    resolve_output_dir,
    write_config_snapshot,
)
from export_dataset import export_dataset
from train_gpt2 import run_training


def _parse_int_list(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _parse_float_list(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def _json_dump(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_root = resolve_output_dir(
        explicit_out_dir=args.out_root,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        version=args.version,
        stage="sweep",
        run_name=None,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(out_root, args, getattr(args, "_config_path", None))

    num_hidden_grid = _parse_int_list(args.num_hidden_grid)
    eps_min_grid = _parse_float_list(args.epsilon_min_grid)
    eps_max_grid = _parse_float_list(args.epsilon_max_grid)
    eps_sched_grid = [s.strip() for s in args.epsilon_schedule_grid.split(",") if s.strip()]
    eta_grid = _parse_float_list(args.eta_grid)
    vocab_grid = _parse_int_list(args.vocab_size_grid)
    dataset_sizes = _parse_int_list(args.dataset_sizes)
    model_sizes = [s.strip() for s in args.model_sizes.split(",") if s.strip()]

    aggregate_path = out_root / "results.jsonl"
    completed = 0
    skipped = 0

    gen_grid = list(
        itertools.product(
            num_hidden_grid,
            eps_min_grid,
            eps_max_grid,
            eps_sched_grid,
            eta_grid,
            vocab_grid,
        )
    )

    for (
        num_hidden,
        eps_min,
        eps_max,
        eps_sched,
        eta,
        vocab_size,
    ) in gen_grid:
        gen_cfg = {
            "num_hidden": num_hidden,
            "epsilon_min": eps_min,
            "epsilon_max": eps_max,
            "epsilon_schedule": eps_sched,
            "eta": eta,
            "vocab_size": vocab_size,
            "q_type": args.q_type,
            "num_clusters": args.num_clusters,
            "cluster_stickiness": args.cluster_stickiness,
            "emission_type": args.emission_type,
            "seed": args.seed,
        }
        dataset_id = _stable_id("dataset", gen_cfg)
        dataset_dir = out_root / "datasets" / dataset_id
        metadata_path = dataset_dir / "metadata.json"

        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            total_tokens = max(
                args.total_tokens,
                int(max(dataset_sizes) / max(1e-8, args.train_frac)) + 2 * args.block_size + 1,
            )
            ds_args = SimpleNamespace(
                out_dir=dataset_dir,
                hmm_path=None,
                num_hidden=num_hidden,
                vocab_size=vocab_size,
                epsilon_min=eps_min,
                epsilon_max=eps_max,
                epsilon_schedule=eps_sched,
                powerlaw_exponent=args.powerlaw_exponent,
                q_type=args.q_type,
                num_clusters=args.num_clusters,
                cluster_stickiness=args.cluster_stickiness,
                emission_type=args.emission_type,
                eta=eta,
                emission_concentration=args.emission_concentration,
                total_tokens=total_tokens,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                max_lag=args.max_lag,
                max_context=args.max_context,
                fit_xmin=args.fit_xmin,
                fit_xmax=args.fit_xmax,
                save_bin=args.save_bin,
                save_hf=args.save_hf,
                seed=args.seed,
            )
            metadata = export_dataset(ds_args)

        mi_slope = metadata["diagnostics"]["mi_fit"]["slope"]
        ce_slope = metadata["diagnostics"]["conditional_entropy_fit"]["slope"]

        for model_size in model_sizes:
            scaling_group_payload = {
                "dataset_id": dataset_id,
                "model_size": model_size,
                "num_hidden": num_hidden,
                "eps_min": eps_min,
                "eps_max": eps_max,
                "eps_sched": eps_sched,
                "eta": eta,
                "vocab_size": vocab_size,
            }
            scaling_group_id = _stable_id("group", scaling_group_payload)

            for n_tokens in dataset_sizes:
                run_payload = {
                    **scaling_group_payload,
                    "dataset_size": n_tokens,
                    "seed": args.seed,
                }
                run_id = _stable_id("run", run_payload)
                run_dir = out_root / "runs" / run_id
                result_path = run_dir / "result.json"

                if result_path.exists():
                    skipped += 1
                    continue

                run_dir.mkdir(parents=True, exist_ok=True)

                train_args = SimpleNamespace(
                    train_tokens=dataset_dir / "train.npy",
                    val_tokens=dataset_dir / "val.npy",
                    out_dir=run_dir,
                    model_preset=model_size,
                    n_layer=6,
                    n_head=4,
                    n_embd=256,
                    vocab_size=vocab_size,
                    block_size=args.block_size,
                    dropout=args.dropout,
                    bias=True,
                    batch_size=args.batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    max_iters=args.max_iters,
                    eval_interval=args.eval_interval,
                    eval_iters=args.eval_iters,
                    learning_rate=args.learning_rate,
                    min_lr=args.min_lr,
                    warmup_iters=args.warmup_iters,
                    lr_decay_iters=args.lr_decay_iters,
                    decay_lr=True,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    weight_decay=args.weight_decay,
                    grad_clip=args.grad_clip,
                    amp=args.amp,
                    cpu=args.cpu,
                    seed=args.seed,
                    train_tokens_limit=n_tokens,
                    overwrite=False,
                )
                result = run_training(train_args)
                result.update(
                    {
                        "run_id": run_id,
                        "dataset_id": dataset_id,
                        "scaling_group_id": scaling_group_id,
                        "dataset_size": int(n_tokens),
                        "model_size": model_size,
                        "num_hidden": int(num_hidden),
                        "epsilon_min": float(eps_min),
                        "epsilon_max": float(eps_max),
                        "epsilon_schedule": eps_sched,
                        "eta": float(eta),
                        "vocab_size": int(vocab_size),
                        "mi_slope": float(mi_slope),
                        "conditional_entropy_slope": float(ce_slope),
                    }
                )
                _json_dump(result_path, result)
                with aggregate_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                completed += 1

    summary = {
        "completed_runs": completed,
        "skipped_existing_runs": skipped,
        "results_file": str(aggregate_path),
    }
    _json_dump(out_root / "sweep_summary.json", summary)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run resumable scaling sweeps")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=Path("experiments"))
    p.add_argument("--experiment-name", type=str, default="pc_hmm_scaling")
    p.add_argument("--version", type=str, default="v001")
    p.add_argument("--out-root", type=Path, default=None)

    p.add_argument("--num-hidden-grid", type=str, default="512")
    p.add_argument("--epsilon-min-grid", type=str, default="1e-4")
    p.add_argument("--epsilon-max-grid", type=str, default="1e-1")
    p.add_argument("--epsilon-schedule-grid", type=str, default="logspace")
    p.add_argument("--eta-grid", type=str, default="0.15")
    p.add_argument("--vocab-size-grid", type=str, default="256")

    p.add_argument("--q-type", type=str, default="uniform")
    p.add_argument("--num-clusters", type=int, default=16)
    p.add_argument("--cluster-stickiness", type=float, default=0.92)
    p.add_argument("--emission-type", type=str, default="peaked")
    p.add_argument("--powerlaw-exponent", type=float, default=1.25)
    p.add_argument("--emission-concentration", type=float, default=50.0)

    p.add_argument("--dataset-sizes", type=str, default="262144,524288,1048576")
    p.add_argument("--model-sizes", type=str, default="small,medium,large")
    p.add_argument("--total-tokens", type=int, default=2**22)
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--val-frac", type=float, default=0.05)

    p.add_argument("--max-lag", type=int, default=128)
    p.add_argument("--max-context", type=int, default=8)
    p.add_argument("--fit-xmin", type=float, default=2)
    p.add_argument("--fit-xmax", type=float, default=64)
    p.add_argument("--save-bin", action="store_true")
    p.add_argument("--save-hf", action="store_true")

    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=40)

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--lr-decay-iters", type=int, default=2000)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    parser = build_argparser()
    _cfg, cfg_path = inject_config_defaults(parser, sys.argv[1:], section="sweep")
    args = parser.parse_args()
    args._config_path = cfg_path
    summary = run_sweep(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
