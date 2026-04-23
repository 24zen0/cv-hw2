# Notebook-to-Scripts Training Commands

This project now runs training through one script:

- `python src/train.py --experiment <id> --config configs/experiments.yaml`

All experiments are selected by `--experiment` ID, so you do not need eight separate scripts.

The training pipeline now uses the local `torchvision.datasets.CIFAR10` copy stored at:

- `/mnt/home/zengyitao/cv-hw2/cifar-10-python`

## 1) Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2) Command Template

Use this template for any run:

```bash
python src/train.py --experiment <experiment_id> --config configs/experiments.yaml
```

## 2.1) Prepare the Reproducible Split File

Generate the default train/validation/test split manifest used by the training config:

```bash
python src/prepare_cifar10_split.py
```

This writes:

- `/mnt/home/zengyitao/cv-hw2/splits/cifar10_train12000_val10_seed42.json`

## 2.2) Explore the Local CIFAR-10 Dataset

Generate dataset summary files, sample grids, and per-class example images:

```bash
python src/explore_cifar10.py
```

This writes outputs under:

- `/mnt/home/zengyitao/cv-hw2/CIFAR-10-data-exploration`

## 3) Commands for Each Experiment

### Baseline and controlled CNN runs

```bash
python src/train.py --experiment baseline --config configs/experiments.yaml
python src/train.py --experiment run_a_deeper --config configs/experiments.yaml
python src/train.py --experiment run_b_kernel5 --config configs/experiments.yaml
python src/train.py --experiment run_c_avgpool --config configs/experiments.yaml
```

### Architecture-family runs

```bash
python src/train.py --experiment alexnet --config configs/experiments.yaml
python src/train.py --experiment vgg11_bn --config configs/experiments.yaml
python src/train.py --experiment inception_v3 --config configs/experiments.yaml
python src/train.py --experiment resnet18 --config configs/experiments.yaml
```

## 4) Useful Optional Flags

Set a custom run name:

```bash
python src/train.py --experiment baseline --config configs/experiments.yaml --run-name my_custom_run
```

Override base output directory:

```bash
python src/train.py --experiment baseline --config configs/experiments.yaml --output-dir /mnt/home/zengyitao/cv-hw2/output
```

Disable Weights & Biases (wandb, short for Weights & Biases):

```bash
python src/train.py --experiment baseline --config configs/experiments.yaml --no-wandb
```

## 5) Output Location

By default, each run is saved under:

- `/mnt/home/zengyitao/cv-hw2/output/<experiment_id>/<run_name>/`

Typical files include:

- `model.pt`
- `config_resolved.yaml`
- `history.json`
- `metrics.json`
- `summary.csv`
- `predictions_test.json`

