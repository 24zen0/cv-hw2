from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml


def init_wandb_run(config: Dict):
    wandb_cfg = config.get("wandb", {})
    if not bool(wandb_cfg.get("enable", False)):
        return None

    try:
        import wandb  # Imported lazily so wandb is optional when disabled.
    except Exception as exc:
        raise RuntimeError("wandb is enabled but the package is unavailable.") from exc

    return wandb.init(
        entity=wandb_cfg.get("entity"),
        project=wandb_cfg.get("project"),
        group=wandb_cfg.get("group"),
        tags=wandb_cfg.get("tags"),
        mode=wandb_cfg.get("mode", "online"),
        name=config["run_name"],
        config=config,
    )


def log_epoch_to_wandb(run, metrics: Dict):
    if run is not None:
        run.log(metrics)


def finish_wandb_run(run, final_metrics: Optional[Dict] = None):
    if run is None:
        return
    if final_metrics:
        run.summary.update(final_metrics)
    run.finish()


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_run_artifacts(
    output_dir: str,
    resolved_config: Dict,
    history: Dict,
    result: Dict,
    best_checkpoint: Dict,
    predictions: Dict,
):
    output = ensure_output_dir(output_dir)

    model_path = output / "model.pt"
    torch.save(best_checkpoint, model_path)

    config_path = output / "config_resolved.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False)

    history_path = output / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    metrics = dict(result)
    metrics["best_monitor"] = best_checkpoint.get("best_monitor")
    metrics["best_monitor_value"] = best_checkpoint.get("best_monitor_value")
    metrics["best_epoch"] = best_checkpoint.get("best_epoch")
    metrics["early_stopped"] = best_checkpoint.get("early_stopped")
    metrics["dataset_limits"] = {
        "train_limit": resolved_config["data"].get("train_limit"),
        "val_limit": resolved_config["data"].get("val_limit"),
        "test_limit": resolved_config["data"].get("test_limit"),
    }

    metrics_path = output / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary_path = output / "summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    predictions_path = output / "predictions_test.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    return {
        "model": str(model_path),
        "config": str(config_path),
        "history": str(history_path),
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "predictions": str(predictions_path),
    }
