from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10


DEFAULT_ROOT = "/mnt/home/zengyitao/cv-hw2/cifar-10-python"
DEFAULT_OUTPUT = "/mnt/home/zengyitao/cv-hw2/CIFAR-10-data-exploration"


def parse_args():
    parser = argparse.ArgumentParser(description="Explore the local CIFAR-10 dataset and export artifacts.")
    parser.add_argument("--root-dir", default=DEFAULT_ROOT, help="Directory containing cifar-10-batches-py")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Directory for exploration outputs")
    parser.add_argument("--samples-per-class", type=int, default=2, help="Number of class images to export")
    return parser.parse_args()


def write_distribution_csv(path: Path, class_names, counts):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "count"])
        for class_idx, class_name in enumerate(class_names):
            writer.writerow([class_name, counts.get(class_idx, 0)])


def save_sample_grid(dataset, class_names, output_path: Path, title: str):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        image, label = dataset[i]
        ax.imshow(image)
        ax.set_title(class_names[label])
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def save_per_class_images(dataset, class_names, output_dir: Path, samples_per_class: int):
    class_hits = defaultdict(int)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        class_name = class_names[label]
        hit_count = class_hits[class_name]
        if hit_count >= samples_per_class:
            continue
        image.save(output_dir / f"{class_name}_{hit_count}.png")
        class_hits[class_name] += 1
        if all(class_hits[name] >= samples_per_class for name in class_names):
            break


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CIFAR10(root=args.root_dir, train=True, download=False)
    test_ds = CIFAR10(root=args.root_dir, train=False, download=False)
    class_names = list(train_ds.classes)

    train_counts = Counter(train_ds.targets)
    test_counts = Counter(test_ds.targets)

    train_array = np.asarray(train_ds.data, dtype=np.float32) / 255.0
    channel_mean = train_array.mean(axis=(0, 1, 2)).tolist()
    channel_std = train_array.std(axis=(0, 1, 2)).tolist()

    dataset_info = {
        "root_dir": args.root_dir,
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "image_shape": list(train_ds.data[0].shape),
        "channel_mean": channel_mean,
        "channel_std": channel_std,
    }

    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    write_distribution_csv(output_dir / "train_class_distribution.csv", class_names, train_counts)
    write_distribution_csv(output_dir / "test_class_distribution.csv", class_names, test_counts)

    save_sample_grid(train_ds, class_names, output_dir / "train_sample_grid.png", "CIFAR-10 Train Samples")
    save_sample_grid(test_ds, class_names, output_dir / "test_sample_grid.png", "CIFAR-10 Test Samples")

    save_per_class_images(train_ds, class_names, output_dir / "train_class_examples", args.samples_per_class)
    save_per_class_images(test_ds, class_names, output_dir / "test_class_examples", args.samples_per_class)

    report_lines = [
        "# CIFAR-10 Exploration",
        "",
        f"- Root directory: `{args.root_dir}`",
        f"- Number of classes: {len(class_names)}",
        f"- Train size: {len(train_ds)}",
        f"- Test size: {len(test_ds)}",
        f"- Image shape: {tuple(train_ds.data[0].shape)}",
        f"- Channel mean: {channel_mean}",
        f"- Channel std: {channel_std}",
    ]
    with open(output_dir / "exploration_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Saved exploration outputs to: {output_dir}")


if __name__ == "__main__":
    main()
