#!/usr/bin/env python3
"""Plot MI curves for multiple experiment versions (no job-id path required)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_mi(stats_json: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(stats_json.read_text())
    lags = np.asarray(d["mi"]["lags"], dtype=float)
    mi = np.asarray(d["mi"]["mi"], dtype=float)
    return lags, mi


def main() -> None:
    p = argparse.ArgumentParser(description="Compare MI curves across versions")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/scratch/gpfs/SARANGG/cw5411/pc_hmm_scaling"),
        help="Root that contains vXXX directories",
    )
    p.add_argument("--versions", type=str, default="v016,v017,v018")
    p.add_argument("--run-name", type=str, default="N4194304")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/scratch/gpfs/SARANGG/cw5411/plots/mi_compare_v016_v017_v018.png"),
    )
    args = p.parse_args()

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]

    plt.figure(figsize=(7, 5), dpi=180)
    for version in versions:
        stats_json = (
            args.root
            / version
            / "datasets_grid"
            / args.run_name
            / "stats"
            / "stats_summary.json"
        )
        lags, mi = load_mi(stats_json)
        plt.semilogy(lags, mi, lw=1.5, label=version)

    plt.xlabel("Lag")
    plt.ylabel("MI I(X_t; X_{t+lag})")
    plt.title("MI Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
