from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10


DEFAULT_ROOT = "/mnt/home/zengyitao/cv-hw2/cifar-10-python"
DEFAULT_OUTPUT = "/mnt/home/zengyitao/cv-hw2/splits/cifar10_train12000_val10_seed42.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a reproducible CIFAR-10 split manifest.")
    parser.add_argument("--root-dir", default=DEFAULT_ROOT, help="Directory containing cifar-10-batches-py")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to write the split manifest JSON")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation fraction from the train split")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for the split")
    parser.add_argument("--train-limit", type=int, default=12000, help="Optional train limit after splitting")
    parser.add_argument("--val-limit", type=int, default=None, help="Optional validation limit after splitting")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional test limit")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified splitting")
    return parser.parse_args()


def apply_limit(indices, limit):
    if limit is None:
        return list(indices)
    return list(indices)[: min(limit, len(indices))]


def class_distribution(indices, targets, class_names):
    counts = Counter(targets[idx] for idx in indices)
    return {class_names[class_idx]: counts.get(class_idx, 0) for class_idx in range(len(class_names))}


def main():
    args = parse_args()
    root_dir = args.root_dir
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = CIFAR10(root=root_dir, train=True, download=False)
    test_ds = CIFAR10(root=root_dir, train=False, download=False)

    all_train_indices = list(range(len(train_ds)))
    stratify_labels = None if args.no_stratify else train_ds.targets
    train_indices, val_indices = train_test_split(
        all_train_indices,
        test_size=args.val_split,
        random_state=args.split_seed,
        stratify=stratify_labels,
    )

    train_indices = apply_limit(train_indices, args.train_limit)
    val_indices = apply_limit(val_indices, args.val_limit)
    test_indices = apply_limit(range(len(test_ds)), args.test_limit)

    manifest = {
        "root_dir": root_dir,
        "class_names": list(train_ds.classes),
        "val_split": args.val_split,
        "split_seed": args.split_seed,
        "stratified": not args.no_stratify,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_indices),
        "train_distribution": class_distribution(train_indices, train_ds.targets, train_ds.classes),
        "val_distribution": class_distribution(val_indices, train_ds.targets, train_ds.classes),
        "test_distribution": class_distribution(test_indices, test_ds.targets, test_ds.classes),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved split manifest to: {output_path}")
    print(
        f"Sizes | train={manifest['train_size']}, "
        f"val={manifest['val_size']}, test={manifest['test_size']}"
    )


if __name__ == "__main__":
    main()
