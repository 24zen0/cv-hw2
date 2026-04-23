from __future__ import annotations

import argparse
from typing import Dict

from config import load_yaml_config, resolve_run_config
from data import build_dataloaders_from_config, get_num_classes
from logging_utils import finish_wandb_run, init_wandb_run, log_epoch_to_wandb, save_run_artifacts
from models import build_model
from train_utils import get_device, run_training, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train one experiment from YAML config.")
    parser.add_argument("--experiment", required=True, help="Experiment id from configs/experiments.yaml")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--run-name", default=None, help="Optional run name")
    parser.add_argument("--output-dir", default=None, help="Optional override for base output directory")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_cfg = load_yaml_config(args.config)
    cfg = resolve_run_config(
        raw_cfg,
        experiment_id=args.experiment,
        run_name=args.run_name,
        output_dir_override=args.output_dir,
        no_wandb=args.no_wandb,
    )

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "auto")))
    print(f"Using device: {device}")
    print(f"Run name: {cfg['run_name']}")
    print(f"Output dir: {cfg['output_dir']}")

    train_loader, val_loader, test_loader, split_info = build_dataloaders_from_config(cfg)
    print(
        "Dataset sizes | "
        f"train={split_info.train_size}, val={split_info.val_size}, test={split_info.test_size}"
    )

    num_classes = int(cfg["data"].get("num_classes", get_num_classes()))
    model = build_model(cfg["experiment_id"], num_classes=num_classes).to(device)

    wandb_run = init_wandb_run(cfg)

    def _epoch_logger(metrics: Dict):
        print(
            f"[{cfg['experiment_id']}] Epoch {metrics['epoch']} | "
            f"train_loss={metrics['train_loss']:.4f}, train_acc={metrics['train_acc']:.4f}, "
            f"val_loss={metrics['val_loss']:.4f}, val_acc={metrics['val_acc']:.4f}, "
            f"val_recall={metrics['val_recall_macro']:.4f}, val_f1={metrics['val_f1_macro']:.4f}"
        )
        log_epoch_to_wandb(wandb_run, metrics)

    result, history, best_checkpoint, predictions = run_training(
        config=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epoch_logger=_epoch_logger,
    )

    result_dict = result.to_dict()
    artifacts = save_run_artifacts(
        output_dir=cfg["output_dir"],
        resolved_config=cfg,
        history=history,
        result=result_dict,
        best_checkpoint=best_checkpoint,
        predictions=predictions,
    )

    finish_wandb_run(
        wandb_run,
        final_metrics={
            "test_accuracy": result_dict["test_accuracy"],
            "test_recall_macro": result_dict["test_recall_macro"],
            "test_f1_macro": result_dict["test_f1_macro"],
            "best_epoch": result_dict["best_epoch"],
            "early_stopped": result_dict["early_stopped"],
        },
    )

    print("Final test metrics:")
    print(
        f"test_acc={result_dict['test_accuracy']:.4f}, "
        f"test_recall_macro={result_dict['test_recall_macro']:.4f}, "
        f"test_f1_macro={result_dict['test_f1_macro']:.4f}"
    )
    print(f"Best epoch by validation monitor: {result_dict['best_epoch']}")
    print("Saved artifacts:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
