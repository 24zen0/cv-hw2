from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = deepcopy(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_default_run_name(experiment_id: str) -> str:
    return f"{datetime.now().strftime('%m-%d-%H-%M-%S')}_{experiment_id}"


def _find_experiment(config: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
    for exp in config.get("experiments", []):
        if exp.get("id") == experiment_id:
            return deepcopy(exp)
    raise ValueError(f"Experiment id '{experiment_id}' not found in config.")


def _build_architecture_note(experiment: Dict[str, Any]) -> str:
    if experiment.get("architecture_note"):
        return str(experiment["architecture_note"])

    arch = experiment.get("architecture", {})
    if not isinstance(arch, dict) or not arch:
        return "No architecture note provided"

    parts = [f"{k}={v}" for k, v in arch.items()]
    return ", ".join(parts)


def resolve_run_config(
    config: Dict[str, Any],
    experiment_id: str,
    run_name: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    no_wandb: bool = False,
) -> Dict[str, Any]:
    project_cfg = deepcopy(config.get("project", {}))
    data_defaults = deepcopy(config.get("data", {}))
    train_defaults = deepcopy(config.get("train", {}))
    wandb_defaults = deepcopy(config.get("wandb", {}))

    experiment = _find_experiment(config, experiment_id)

    resolved_data = _deep_merge(data_defaults, experiment.get("data_override"))
    resolved_train = _deep_merge(train_defaults, experiment.get("train_override"))
    resolved_wandb = _deep_merge(wandb_defaults, experiment.get("wandb_override"))

    final_run_name = run_name or build_default_run_name(experiment_id)

    base_output_dir = output_dir_override or project_cfg.get("output_dir", "output")
    run_output_dir = str(Path(base_output_dir) / experiment_id / final_run_name)

    if no_wandb:
        resolved_wandb["enable"] = False

    early_stopping_defaults = {
        "enable": True,
        "monitor": "val_loss",
        "mode": "min",
        "patience": 5,
        "min_delta": 0.001,
    }
    early_stopping_cfg = _deep_merge(early_stopping_defaults, resolved_train.get("early_stopping"))

    resolved = {
        "project": project_cfg,
        "experiment_id": experiment_id,
        "name": experiment.get("name", experiment_id),
        "run_name": final_run_name,
        "model_family": experiment.get("model_family"),
        "architecture": deepcopy(experiment.get("architecture", {})),
        "architecture_note": _build_architecture_note(experiment),
        "data": resolved_data,
        "train": resolved_train,
        "wandb": resolved_wandb,
        "output_dir": run_output_dir,
        "early_stopping": early_stopping_cfg,
    }

    # Common flattened fields for convenience in downstream code.
    resolved["seed"] = project_cfg.get("seed", 42)
    resolved["device"] = project_cfg.get("device", "auto")

    return resolved
