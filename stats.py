"""Information-theoretic and spectral diagnostics for token sequences and HMMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def _safe_entropy_from_counts(counts: NDArray[np.int64]) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log(probs)).sum())


def estimate_pairwise_mi(
    tokens: NDArray[np.int64], max_lag: int, vocab_size: int
) -> dict[str, NDArray[np.float64]]:
    """Empirical pairwise MI I(X_t; X_{t+lag}) for lag=1..max_lag."""
    x = np.asarray(tokens, dtype=np.int64)
    lags = np.arange(1, max_lag + 1, dtype=np.int64)
    mi = np.zeros_like(lags, dtype=np.float64)

    for idx, lag in enumerate(lags):
        a = x[:-lag]
        b = x[lag:]
        joint_idx = a * vocab_size + b
        joint_counts = np.bincount(joint_idx, minlength=vocab_size * vocab_size).reshape(
            vocab_size, vocab_size
        )
        n = joint_counts.sum()
        if n == 0:
            mi[idx] = 0.0
            continue
        pxy = joint_counts / n
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        nz = pxy > 0
        mi[idx] = float(np.sum(pxy[nz] * (np.log(pxy[nz]) - np.log((px @ py)[nz]))))

    return {"lags": lags.astype(np.float64), "mi": mi}


def estimate_conditional_entropy(
    tokens: NDArray[np.int64],
    max_context: int,
    vocab_size: int,
    method: str = "ngram",
) -> dict[str, NDArray[np.float64]]:
    """Estimate H(X_t | X_{t-k:t-1}) for k=0..max_context using exact n-grams."""
    if method != "ngram":
        raise ValueError("Only method='ngram' is currently implemented")

    x = np.asarray(tokens, dtype=np.int64)
    ks = np.arange(0, max_context + 1, dtype=np.int64)
    hvals = np.zeros_like(ks, dtype=np.float64)

    unigram = np.bincount(x, minlength=vocab_size)
    hvals[0] = _safe_entropy_from_counts(unigram)

    for i, k in enumerate(ks[1:], start=1):
        if len(x) <= k:
            hvals[i] = np.nan
            continue
        contexts: dict[tuple[int, ...], NDArray[np.int64]] = {}
        for t in range(k, len(x)):
            ctx = tuple(int(v) for v in x[t - k : t])
            nxt = int(x[t])
            if ctx not in contexts:
                contexts[ctx] = np.zeros(vocab_size, dtype=np.int64)
            contexts[ctx][nxt] += 1

        total = 0
        h_cond = 0.0
        for counts in contexts.values():
            c = int(counts.sum())
            total += c
            h_cond += c * _safe_entropy_from_counts(counts)
        hvals[i] = h_cond / max(total, 1)

    return {"context": ks.astype(np.float64), "H": hvals}


def estimate_autocorrelation(
    tokens: NDArray[np.int64], max_lag: int
) -> dict[str, NDArray[np.float64]]:
    """Estimate scalar autocorrelation of token IDs treated as a time series."""
    x = np.asarray(tokens, dtype=np.float64)
    x = x - x.mean()
    var = x.var()
    lags = np.arange(1, max_lag + 1)
    ac = np.zeros_like(lags, dtype=np.float64)

    if var <= 0:
        return {"lags": lags.astype(np.float64), "autocorr": ac}

    for i, lag in enumerate(lags):
        ac[i] = float(np.dot(x[:-lag], x[lag:]) / ((len(x) - lag) * var))

    return {"lags": lags.astype(np.float64), "autocorr": ac}


def estimate_transition_spectrum(P: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """Compute transition matrix eigenvalue and singular value spectra."""
    eigvals = np.linalg.eigvals(P)
    svals = np.linalg.svd(P, compute_uv=False)
    eigvals_sorted = eigvals[np.argsort(-np.abs(eigvals))]
    svals_sorted = np.sort(svals)[::-1]
    return {
        "eigenvalues": eigvals_sorted,
        "abs_eigenvalues": np.abs(eigvals_sorted),
        "singular_values": svals_sorted,
    }


def fit_effective_powerlaw(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xmin: float | None = None,
    xmax: float | None = None,
) -> dict[str, Any]:
    """Fit y ~ c * x^slope over a selected log-log window."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(xx) & np.isfinite(yy) & (yy > 0)
    if xmin is not None:
        mask &= xx >= xmin
    if xmax is not None:
        mask &= xx <= xmax

    x_fit = xx[mask]
    y_fit = yy[mask]
    if len(x_fit) < 2:
        return {
            "success": False,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "x_fit": x_fit,
            "y_fit": y_fit,
            "y_pred": np.full_like(y_fit, np.nan),
        }

    lx = np.log(x_fit)
    ly = np.log(y_fit)
    slope, intercept = np.polyfit(lx, ly, 1)
    ly_pred = slope * lx + intercept
    ss_res = float(np.sum((ly - ly_pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "success": True,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "x_fit": x_fit,
        "y_fit": y_fit,
        "y_pred": np.exp(ly_pred),
    }


def plot_mi_curve(
    lags: NDArray[np.float64],
    mi: NDArray[np.float64],
    fit_summary: dict[str, Any] | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.loglog(lags, mi, marker="o", lw=1.5, label="MI")
    if fit_summary and fit_summary.get("success", False):
        ax.loglog(
            fit_summary["x_fit"],
            fit_summary["y_pred"],
            "--",
            label=f"fit slope={fit_summary['slope']:.3f}",
        )
    ax.set_xlabel("Lag")
    ax.set_ylabel("I(X_t;X_{t+lag})")
    ax.set_title("Pairwise MI vs Lag")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_conditional_entropy_curve(
    context: NDArray[np.float64],
    H: NDArray[np.float64],
    fit_summary: dict[str, Any] | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(context, H, marker="o", lw=1.5, label="H(X_t|context)")
    if fit_summary and fit_summary.get("success", False):
        ax.loglog(
            fit_summary["x_fit"],
            fit_summary["y_pred"],
            "--",
            label=f"gap fit slope={fit_summary['slope']:.3f}",
        )
    ax.set_xlabel("Context length k")
    ax.set_ylabel("Conditional entropy")
    ax.set_title("Conditional Entropy vs Context")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_spectrum(
    spectrum_values: NDArray[np.float64],
    save_path: str | Path | None = None,
    title: str = "Transition Spectrum",
) -> tuple[plt.Figure, plt.Axes]:
    idx = np.arange(1, len(spectrum_values) + 1)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.semilogy(idx, np.abs(spectrum_values), marker=".", lw=1.0)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute basic sequence diagnostics")
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--max-lag", type=int, default=128)
    p.add_argument("--max-context", type=int, default=8)
    p.add_argument("--fit-xmin", type=float, default=2)
    p.add_argument("--fit-xmax", type=float, default=64)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/stats"))
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokens = np.load(args.tokens)
    mi = estimate_pairwise_mi(tokens, args.max_lag, args.vocab_size)
    mi_fit = fit_effective_powerlaw(mi["lags"], mi["mi"], args.fit_xmin, args.fit_xmax)

    ce = estimate_conditional_entropy(tokens, args.max_context, args.vocab_size)
    H = ce["H"]
    k = ce["context"]
    gap = H[:-1] - H[-1]
    k_gap = k[:-1]
    valid = k_gap >= 1
    ce_fit = fit_effective_powerlaw(k_gap[valid], gap[valid], 1, args.max_context - 1)

    plot_mi_curve(mi["lags"], mi["mi"], mi_fit, args.out_dir / "mi_curve.png")
    plot_conditional_entropy_curve(k, H, ce_fit, args.out_dir / "conditional_entropy_curve.png")

    out = {
        "mi": {"lags": mi["lags"].tolist(), "mi": mi["mi"].tolist()},
        "conditional_entropy": {"context": k.tolist(), "H": H.tolist()},
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
    with (args.out_dir / "stats_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
