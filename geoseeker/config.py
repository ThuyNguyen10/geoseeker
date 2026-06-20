"""Central configuration and shared constants for GeoSeeker.

All paths are resolved relative to the project root so the scripts work no
matter what the current working directory is.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "geoseeker_best.pt"

# ImageNet normalisation (used by all torchvision pretrained weights).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    """Hyper-parameters and runtime options for training.

    These defaults are tuned to be *CPU friendly*: a small input size, a
    light backbone, and the option to freeze the backbone so only the
    classification head is trained (which is very fast).
    """

    # Data
    dataset_dir: Path = DATASET_DIR
    image_size: int = 224
    # Drop countries that have fewer than this many images. Stops the model
    # from being asked to learn a class from a single example.
    min_images_per_class: int = 20
    val_split: float = 0.15
    # Cap images per class to fight the heavy imbalance (e.g. US has 12k).
    max_images_per_class: int = 1500

    # Model
    backbone: str = "resnet18"  # resnet18 | resnet50 | mobilenet_v3_large
    pretrained: bool = True
    freeze_backbone: bool = False

    # Optimisation
    epochs: int = 15
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1

    # Runtime
    num_workers: int = field(default_factory=lambda: min(8, os.cpu_count() or 2))
    seed: int = 42
    # "auto" picks cuda if available, else cpu.
    device: str = "auto"
    output: Path = BEST_MODEL_PATH
