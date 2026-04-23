from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


DEFAULT_CIFAR10_ROOT = "/mnt/home/zengyitao/cv-hw2/cifar-10-python"


def _get_dataset_root(root_dir: Optional[str] = None) -> str:
    return root_dir or DEFAULT_CIFAR10_ROOT


def _build_base_dataset(root_dir: Optional[str], train: bool) -> CIFAR10:
    root = _get_dataset_root(root_dir)
    batches_dir = Path(root) / "cifar-10-batches-py"
    if not batches_dir.exists():
        raise FileNotFoundError(
            f"Expected CIFAR-10 files under '{batches_dir}', but that directory does not exist."
        )
    return CIFAR10(root=root, train=train, download=False)


def get_num_classes(root_dir: Optional[str] = None) -> int:
    ds = _build_base_dataset(root_dir, train=True)
    return len(ds.classes)


def get_class_names(root_dir: Optional[str] = None) -> List[str]:
    ds = _build_base_dataset(root_dir, train=True)
    return list(ds.classes)


class IndexedCIFAR10Dataset(Dataset):
    def __init__(self, base_dataset: CIFAR10, indices: Sequence[int], transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(
    image_size: int = 32,
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616),
):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, eval_tf


@dataclass
class SplitInfo:
    train_size: int
    val_size: int
    test_size: int


def _apply_limit(indices: Sequence[int], limit: Optional[int]) -> List[int]:
    indices = list(indices)
    if limit is None:
        return indices
    limit = min(int(limit), len(indices))
    return indices[:limit]


def _compute_split_indices(
    targets: Sequence[int],
    val_split: float,
    split_seed: int,
    stratified: bool,
) -> Tuple[List[int], List[int]]:
    all_indices = list(range(len(targets)))
    stratify_labels = list(targets) if stratified else None
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=val_split,
        random_state=split_seed,
        stratify=stratify_labels,
    )
    return list(train_indices), list(val_indices)


def _load_split_manifest(split_file: str) -> Dict:
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders_from_config(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, SplitInfo]:
    data_cfg = config["data"]
    train_cfg = config["train"]

    image_size = int(data_cfg.get("image_size", 32))
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 0))
    val_split = float(data_cfg.get("val_split", 0.1))
    split_seed = int(data_cfg.get("split_seed", config.get("seed", 42)))
    stratified = bool(data_cfg.get("stratified_split", True))

    normalization = data_cfg.get("normalization", {})
    mean = tuple(normalization.get("mean", [0.4914, 0.4822, 0.4465]))
    std = tuple(normalization.get("std", [0.2470, 0.2435, 0.2616]))

    train_tf, eval_tf = build_transforms(image_size=image_size, mean=mean, std=std)

    root_dir = data_cfg.get("root_dir", DEFAULT_CIFAR10_ROOT)
    split_file = data_cfg.get("split_file")
    base_train = _build_base_dataset(root_dir=root_dir, train=True)
    base_test = _build_base_dataset(root_dir=root_dir, train=False)

    if split_file:
        manifest = _load_split_manifest(split_file)
        train_indices = list(manifest["train_indices"])
        val_indices = list(manifest["val_indices"])
        test_indices = list(manifest["test_indices"])
    else:
        train_indices, val_indices = _compute_split_indices(
            targets=base_train.targets,
            val_split=val_split,
            split_seed=split_seed,
            stratified=stratified,
        )
        test_indices = list(range(len(base_test.targets)))
        train_indices = _apply_limit(train_indices, data_cfg.get("train_limit"))
        val_indices = _apply_limit(val_indices, data_cfg.get("val_limit"))
        test_indices = _apply_limit(test_indices, data_cfg.get("test_limit"))

    train_ds = IndexedCIFAR10Dataset(base_train, train_indices, transform=train_tf)
    val_ds = IndexedCIFAR10Dataset(base_train, val_indices, transform=eval_tf)
    test_ds = IndexedCIFAR10Dataset(base_test, test_indices, transform=eval_tf)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    split_info = SplitInfo(
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
    )
    return train_loader, val_loader, test_loader, split_info
