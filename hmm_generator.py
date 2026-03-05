"""Pseudocritical finite-state HMM generator.

This module implements a large finite-state HMM with a dense set of slow
metastable timescales to approximate critical-like behavior over finite lags.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig


@dataclass
class HMMConfig:
    num_hidden: int = 512
    vocab_size: int = 256
    epsilon_min: float = 1e-4
    epsilon_max: float = 1e-1
    epsilon_schedule: str = "logspace"  # logspace | powerlaw | custom
    powerlaw_exponent: float = 1.25
    epsilon_custom: list[float] | None = None
    q_type: str = "uniform"  # uniform | cluster | random
    num_clusters: int = 16
    cluster_stickiness: float = 0.92
    emission_type: str = "peaked"  # onehot | peaked | dirichlet_cluster
    eta: float = 0.15
    emission_concentration: float = 50.0
    seed: int = 0
    dtype: str = "float64"
    extra: dict[str, Any] = field(default_factory=dict)


class PseudoCriticalHMM:
    """Large finite-state HMM with metastable escape-rate parameterization."""

    def __init__(self, config: HMMConfig):
        self.config = config
        self.num_hidden = config.num_hidden
        self.vocab_size = config.vocab_size
        self.dtype = np.float64 if config.dtype == "float64" else np.float32

        rng = np.random.default_rng(config.seed)
        self.epsilon = self._build_epsilon(config)
        self.Q = self._build_mixing_matrix(config, rng)
        self.P = self._build_transition(self.epsilon, self.Q)
        self.O = self._build_emission(config, rng)

        self._validate()

    def _build_epsilon(self, cfg: HMMConfig) -> NDArray[np.float64]:
        n = cfg.num_hidden
        if cfg.epsilon_schedule == "logspace":
            eps = np.logspace(np.log10(cfg.epsilon_max), np.log10(cfg.epsilon_min), n)
        elif cfg.epsilon_schedule == "powerlaw":
            ranks = np.arange(1, n + 1, dtype=np.float64)
            raw = ranks ** (-cfg.powerlaw_exponent)
            raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
            eps = cfg.epsilon_min + (cfg.epsilon_max - cfg.epsilon_min) * raw
        elif cfg.epsilon_schedule == "custom":
            if cfg.epsilon_custom is None or len(cfg.epsilon_custom) != n:
                raise ValueError("epsilon_custom must have length num_hidden")
            eps = np.asarray(cfg.epsilon_custom, dtype=np.float64)
        else:
            raise ValueError(f"Unknown epsilon_schedule={cfg.epsilon_schedule}")

        eps = np.clip(eps, 1e-12, 1.0 - 1e-12)
        return eps.astype(self.dtype)

    def _build_mixing_matrix(
        self, cfg: HMMConfig, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        n = cfg.num_hidden
        Q = np.zeros((n, n), dtype=self.dtype)

        if cfg.q_type == "uniform":
            Q[:] = 1.0 / (n - 1)
            np.fill_diagonal(Q, 0.0)
        elif cfg.q_type == "cluster":
            clusters = np.arange(n) % max(1, cfg.num_clusters)
            for i in range(n):
                same = np.where(clusters == clusters[i])[0]
                other = np.where(clusters != clusters[i])[0]
                same = same[same != i]
                if len(same) > 0:
                    Q[i, same] = cfg.cluster_stickiness / len(same)
                if len(other) > 0:
                    Q[i, other] = (1.0 - cfg.cluster_stickiness) / len(other)
                Q[i, i] = 0.0
        elif cfg.q_type == "random":
            alpha = np.ones(n, dtype=self.dtype)
            for i in range(n):
                row = rng.dirichlet(alpha)
                row[i] = 0.0
                row_sum = row.sum()
                if row_sum <= 0:
                    row[:] = 1.0 / (n - 1)
                    row[i] = 0.0
                else:
                    row /= row_sum
                Q[i] = row
        else:
            raise ValueError(f"Unknown q_type={cfg.q_type}")

        row_sums = Q.sum(axis=1, keepdims=True)
        Q = np.divide(Q, row_sums, out=np.zeros_like(Q), where=row_sums > 0)
        return Q.astype(self.dtype)

    def _build_transition(
        self, epsilon: NDArray[np.float64], Q: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        n = self.num_hidden
        P = epsilon[:, None] * Q
        np.fill_diagonal(P, 1.0 - epsilon)
        return P.astype(self.dtype)

    def _build_emission(
        self, cfg: HMMConfig, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        n, v = cfg.num_hidden, cfg.vocab_size
        O = np.zeros((n, v), dtype=self.dtype)

        if cfg.emission_type == "onehot":
            for z in range(n):
                O[z, z % v] = 1.0
        elif cfg.emission_type == "peaked":
            eta = float(np.clip(cfg.eta, 0.0, 1.0))
            base = eta / max(1, v - 1)
            O[:] = base
            for z in range(n):
                tok = z % v
                O[z, tok] = 1.0 - eta
        elif cfg.emission_type == "dirichlet_cluster":
            c = max(1, cfg.num_clusters)
            cluster_ids = np.arange(n) % c
            prototypes = np.zeros((c, v), dtype=self.dtype)
            for k in range(c):
                center = (k * v) // c
                proto = np.full(v, cfg.eta / max(1, v - 1), dtype=self.dtype)
                proto[center % v] = 1.0 - cfg.eta
                prototypes[k] = proto / proto.sum()
            alpha0 = float(max(1e-3, cfg.emission_concentration))
            for z in range(n):
                alpha = alpha0 * prototypes[cluster_ids[z]] + 1e-3
                O[z] = rng.dirichlet(alpha)
        else:
            raise ValueError(f"Unknown emission_type={cfg.emission_type}")

        O /= O.sum(axis=1, keepdims=True)
        return O.astype(self.dtype)

    def _validate(self) -> None:
        for name, M in [("P", self.P), ("O", self.O), ("Q", self.Q)]:
            if np.any(M < -1e-9):
                raise ValueError(f"{name} has negative entries")
        if not np.allclose(self.P.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Transition matrix rows do not sum to 1")
        if not np.allclose(self.O.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Emission matrix rows do not sum to 1")

    def stationary_distribution(self) -> NDArray[np.float64]:
        """Return stationary distribution pi over hidden states."""
        vals, vecs = eig(self.P.T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        vec = np.real(vecs[:, idx])
        vec = np.maximum(vec, 0.0)
        if vec.sum() <= 0:
            vec = np.ones(self.num_hidden, dtype=self.dtype)
        pi = vec / vec.sum()
        return pi.astype(self.dtype)

    def sample_hidden(self, T: int, seed: int | None = None) -> NDArray[np.int64]:
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        states = np.zeros(T, dtype=np.int64)
        pi = self.stationary_distribution()
        states[0] = rng.choice(self.num_hidden, p=pi)
        for t in range(1, T):
            states[t] = rng.choice(self.num_hidden, p=self.P[states[t - 1]])
        return states

    def sample_observed(self, T: int, seed: int | None = None) -> NDArray[np.int64]:
        hidden = self.sample_hidden(T, seed=seed)
        rng = np.random.default_rng((self.config.seed if seed is None else seed) + 1)
        obs = np.zeros(T, dtype=np.int64)
        for t in range(T):
            obs[t] = rng.choice(self.vocab_size, p=self.O[hidden[t]])
        return obs

    def sample_tokens(self, T: int, seed: int | None = None) -> NDArray[np.int64]:
        return self.sample_observed(T, seed=seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "epsilon": self.epsilon.tolist(),
            "P": self.P.tolist(),
            "Q": self.Q.tolist(),
            "O": self.O.tolist(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            config_json=json.dumps(asdict(self.config)),
            epsilon=self.epsilon,
            P=self.P,
            Q=self.Q,
            O=self.O,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PseudoCriticalHMM":
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        cfg = HMMConfig(**json.loads(data["config_json"].item()))
        obj = cls(cfg)
        obj.epsilon = data["epsilon"]
        obj.P = data["P"]
        obj.Q = data["Q"]
        obj.O = data["O"]
        obj._validate()
        return obj


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build and sample a pseudocritical HMM")
    p.add_argument("--out", type=Path, default=Path("artifacts/hmm_model.npz"))
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-len", type=int, default=0)
    p.add_argument("--sample-out", type=Path, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
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
    hmm.save(args.out)
    print(f"Saved HMM to {args.out}")

    if args.sample_len > 0:
        tokens = hmm.sample_tokens(args.sample_len, seed=args.seed)
        sample_out = args.sample_out or args.out.with_suffix(".sample.npy")
        np.save(sample_out, tokens)
        print(f"Saved sample tokens ({len(tokens)}) to {sample_out}")


if __name__ == "__main__":
    main()
