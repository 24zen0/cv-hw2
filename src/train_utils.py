from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score


@dataclass
class ExperimentResult:
    name: str
    model_family: str
    architecture_note: str
    train_time_sec: float
    best_epoch: int
    train_accuracy: float
    test_accuracy: float
    test_recall_macro: float
    test_f1_macro: float
    early_stopped: bool
    best_monitor_value: float

    def to_dict(self) -> Dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _build_optimizer(model: nn.Module, train_cfg: Dict):
    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_scheduler(optimizer, train_cfg: Dict):
    scheduler_cfg = train_cfg.get("scheduler")
    if not scheduler_cfg:
        return None
    scheduler_type = str(scheduler_cfg.get("type", "")).lower()
    if scheduler_type == "steplr":
        step_size = int(scheduler_cfg.get("step_size", 5))
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if scheduler_type == "cosineannealinglr":
        t_max = int(scheduler_cfg.get("t_max", train_cfg.get("epochs", 10)))
        eta_min = float(scheduler_cfg.get("eta_min", 0.0))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, recall, f1, np.array(all_labels), np.array(all_preds)


def _is_improved(current: float, best: Optional[float], mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < (best - min_delta)
    if mode == "max":
        return current > (best + min_delta)
    raise ValueError(f"Unsupported early-stopping mode: {mode}")


def run_training(
    config: Dict,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    epoch_logger=None,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, config["train"])
    scheduler = _build_scheduler(optimizer, config["train"])

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_recall_macro": [],
        "val_f1_macro": [],
        "learning_rate": [],
    }

    early_cfg = config["early_stopping"]
    early_enable = bool(early_cfg.get("enable", True))
    early_monitor = str(early_cfg.get("monitor", "val_loss"))
    early_mode = str(early_cfg.get("mode", "min")).lower()
    patience = int(early_cfg.get("patience", 5))
    min_delta = float(early_cfg.get("min_delta", 0.001))

    best_metric = None
    best_epoch = 0
    best_state_dict = None
    best_optimizer_state = None
    non_improve_epochs = 0
    early_stopped = False

    epochs = int(config["train"].get("epochs", 10))
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    wall_start = torch.tensor(0.0)  # placeholder for type consistency
    import time

    wall_start_t = time.time()
    if start_time is not None:
        start_time.record()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_recall, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_recall_macro"].append(val_recall)
        history["val_f1_macro"].append(val_f1)
        # Learning rate (L.R.) used during this epoch, before the scheduler's end-of-epoch update.
        lr = float(optimizer.param_groups[0]["lr"])
        history["learning_rate"].append(lr)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_recall_macro": val_recall,
            "val_f1_macro": val_f1,
            "learning_rate": lr,
        }

        if epoch_logger is not None:
            epoch_logger(epoch_metrics)

        monitor_value = epoch_metrics.get(early_monitor)
        if monitor_value is None:
            raise ValueError(f"Early-stopping monitor '{early_monitor}' is unavailable in metrics.")

        if _is_improved(float(monitor_value), best_metric, early_mode, min_delta):
            best_metric = float(monitor_value)
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_optimizer_state = optimizer.state_dict()
            non_improve_epochs = 0
        else:
            non_improve_epochs += 1

        if scheduler is not None:
            scheduler.step()

        if early_enable and non_improve_epochs >= patience:
            early_stopped = True
            break

    if end_time is not None:
        end_time.record()
        torch.cuda.synchronize()
        elapsed_sec = start_time.elapsed_time(end_time) / 1000.0
    else:
        elapsed_sec = time.time() - wall_start_t

    if best_state_dict is None:
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_optimizer_state = optimizer.state_dict()
        best_metric = float(history["val_loss"][-1]) if history["val_loss"] else 0.0
        best_epoch = len(history["val_loss"]) if history["val_loss"] else 0

    model.load_state_dict(best_state_dict)

    _, final_train_acc, _, _, _, _ = evaluate(model, train_loader, criterion, device)
    _, test_acc, test_recall, test_f1, y_true_test, y_pred_test = evaluate(model, test_loader, criterion, device)

    result = ExperimentResult(
        name=config["name"],
        model_family=config["model_family"],
        architecture_note=config["architecture_note"],
        train_time_sec=elapsed_sec,
        best_epoch=best_epoch,
        train_accuracy=final_train_acc,
        test_accuracy=test_acc,
        test_recall_macro=test_recall,
        test_f1_macro=test_f1,
        early_stopped=early_stopped,
        best_monitor_value=float(best_metric),
    )

    best_checkpoint = {
        "model_state_dict": best_state_dict,
        "optimizer_state_dict": best_optimizer_state,
        "best_epoch": best_epoch,
        "best_monitor": early_monitor,
        "best_monitor_value": float(best_metric),
        "early_stopped": early_stopped,
    }

    predictions = {
        "y_true_test": y_true_test.tolist(),
        "y_pred_test": y_pred_test.tolist(),
    }

    return result, history, best_checkpoint, predictions
