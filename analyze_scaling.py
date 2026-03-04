"""Analyze scaling laws from sweep logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from experiment_utils import (
    inject_config_defaults,
    resolve_output_dir,
    write_config_snapshot,
)

def scaling_fn(N: np.ndarray, L_inf: float, A: float, alpha: float) -> np.ndarray:
    return L_inf + A * np.power(N, -alpha)


def fit_scaling_law(df: pd.DataFrame) -> dict[str, Any]:
    x = df["dataset_size"].to_numpy(dtype=float)
    y = df["best_val_loss"].to_numpy(dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]

    p0 = [float(np.min(y) * 0.9), float(np.max(y) - np.min(y) + 1e-4), 0.1]
    bounds = ([0.0, 0.0, 0.0], [10.0, 100.0, 5.0])

    try:
        popt, pcov = curve_fit(
            scaling_fn,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
        perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        ci95 = 1.96 * perr
        yhat = scaling_fn(x, *popt)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            "success": True,
            "L_inf": float(popt[0]),
            "A": float(popt[1]),
            "alpha": float(popt[2]),
            "L_inf_ci95": float(ci95[0]),
            "A_ci95": float(ci95[1]),
            "alpha_ci95": float(ci95[2]),
            "r2": float(r2),
            "x": x,
            "y": y,
            "yhat": yhat,
        }
    except Exception:
        return {
            "success": False,
            "L_inf": np.nan,
            "A": np.nan,
            "alpha": np.nan,
            "L_inf_ci95": np.nan,
            "A_ci95": np.nan,
            "alpha_ci95": np.nan,
            "r2": np.nan,
            "x": x,
            "y": y,
            "yhat": np.full_like(y, np.nan),
        }


def load_results(experiment_root: Path) -> pd.DataFrame:
    aggregate = experiment_root / "results.jsonl"
    rows: list[dict[str, Any]] = []

    if aggregate.exists():
        with aggregate.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        for path in (experiment_root / "runs").glob("*/result.json"):
            with path.open("r", encoding="utf-8") as f:
                rows.append(json.load(f))

    if not rows:
        raise FileNotFoundError("No run results found")
    return pd.DataFrame(rows)


def plot_loss_curves(
    fit_rows: pd.DataFrame, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=220)

    for _, row in fit_rows.iterrows():
        if not bool(row["success"]):
            continue
        x = np.asarray(row["x"], dtype=float)
        y = np.asarray(row["y"], dtype=float)
        yhat = np.asarray(row["yhat"], dtype=float)
        label = f"{row['scaling_group_id']} alpha={row['alpha']:.3f}"
        ax.loglog(x, y, "o", alpha=0.75)
        ax.loglog(x, yhat, "-", lw=1.8, label=label)

    ax.set_xlabel("Dataset size N")
    ax.set_ylabel("Validation loss")
    ax.set_title("Loss vs Dataset Size with Scaling Fits")
    ax.grid(alpha=0.3)
    if len(ax.lines) > 0:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def plot_alpha_correlation(
    fit_rows: pd.DataFrame,
    xcol: str,
    xlabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=220)
    x = fit_rows[xcol].to_numpy(dtype=float)
    y = fit_rows["alpha"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    ax.scatter(x, y, s=28, alpha=0.85)

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        yy = slope * xx + intercept
        ax.plot(xx, yy, "--", lw=1.5, label=f"slope={slope:.3f}")
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Scaling exponent alpha")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def analyze(experiment_root: Path, out_dir: Path, min_points: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(experiment_root)

    group_cols = ["scaling_group_id"]
    fit_records: list[dict[str, Any]] = []

    for gid, gdf in df.groupby(group_cols):
        if len(gdf) < min_points:
            continue
        fit = fit_scaling_law(gdf)
        rec = {
            "scaling_group_id": gid,
            "success": fit["success"],
            "L_inf": fit["L_inf"],
            "A": fit["A"],
            "alpha": fit["alpha"],
            "L_inf_ci95": fit["L_inf_ci95"],
            "A_ci95": fit["A_ci95"],
            "alpha_ci95": fit["alpha_ci95"],
            "r2": fit["r2"],
            "x": fit["x"].tolist(),
            "y": fit["y"].tolist(),
            "yhat": fit["yhat"].tolist(),
            "model_size": str(gdf.iloc[0].get("model_size", "unknown")),
            "mi_slope": float(gdf.iloc[0].get("mi_slope", np.nan)),
            "conditional_entropy_slope": float(
                gdf.iloc[0].get("conditional_entropy_slope", np.nan)
            ),
            "num_points": int(len(gdf)),
        }
        fit_records.append(rec)

    fit_df = pd.DataFrame(fit_records)
    if fit_df.empty:
        raise RuntimeError("No groups had enough points to fit")

    fit_df.to_csv(out_dir / "scaling_fit_summary.csv", index=False)
    with (out_dir / "scaling_fit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(fit_records, f, indent=2)

    plot_loss_curves(fit_df, out_dir / "loss_vs_dataset_size.png")
    plot_alpha_correlation(
        fit_df,
        xcol="mi_slope",
        xlabel="Measured MI slope",
        out_path=out_dir / "alpha_vs_mi_slope.png",
    )
    plot_alpha_correlation(
        fit_df,
        xcol="conditional_entropy_slope",
        xlabel="Measured conditional-entropy slope",
        out_path=out_dir / "alpha_vs_conditional_entropy_slope.png",
    )

    return {
        "num_runs": int(len(df)),
        "num_fit_groups": int(len(fit_df)),
        "output_dir": str(out_dir),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze scaling curves from sweep output")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=Path("/scratch/gpfs/SARANGG/cw5411"))
    p.add_argument("--experiment-name", type=str, default="pc_hmm_scaling")
    p.add_argument("--version", type=str, default="v001")
    p.add_argument("--experiment-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--min-points", type=int, default=3)
    return p


def main() -> None:
    parser = build_argparser()
    _cfg, cfg_path = inject_config_defaults(parser, sys.argv[1:], section="analysis")
    args = parser.parse_args()

    experiment_root = args.experiment_root or resolve_output_dir(
        explicit_out_dir=None,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        version=args.version,
        stage="sweep",
        run_name=None,
    )
    out_dir = resolve_output_dir(
        explicit_out_dir=args.out_dir,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        version=args.version,
        stage="analysis",
        run_name=None,
    )
    out = analyze(experiment_root, out_dir, args.min_points)
    write_config_snapshot(out_dir, args, cfg_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
