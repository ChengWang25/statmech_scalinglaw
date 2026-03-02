"""Shared utilities for config-driven, versioned experiment management."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, argparse.Namespace):
        return {k: to_jsonable(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def load_json_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Config file must be a JSON object: {cfg_path}")
    return obj


def section_defaults(config: dict[str, Any], section: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for key in ("output_root", "experiment_name", "version"):
        if key in config:
            defaults[key] = config[key]
    block = config.get(section, {})
    if isinstance(block, dict):
        defaults.update(block)
    return defaults


def inject_config_defaults(
    parser: argparse.ArgumentParser, argv: list[str], section: str
) -> tuple[dict[str, Any], Path | None]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    cfg: dict[str, Any] = {}
    cfg_path: Path | None = pre_args.config
    if cfg_path is not None:
        cfg = load_json_config(cfg_path)
        parser.set_defaults(**section_defaults(cfg, section))
    return cfg, cfg_path


def resolve_output_dir(
    explicit_out_dir: str | Path | None,
    output_root: str | Path,
    experiment_name: str,
    version: str,
    stage: str,
    run_name: str | None = None,
) -> Path:
    if explicit_out_dir is not None:
        return Path(explicit_out_dir)
    out = Path(output_root) / experiment_name / version / stage
    if run_name:
        out = out / run_name
    return out


def write_config_snapshot(
    out_dir: str | Path,
    args_namespace: argparse.Namespace,
    cfg_path: Path | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "resolved_args": to_jsonable(args_namespace),
        "config_path": str(cfg_path) if cfg_path else None,
    }
    with (out / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
