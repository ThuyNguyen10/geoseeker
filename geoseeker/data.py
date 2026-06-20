"""Dataset discovery, filtering, splitting and transforms.

The on-disk layout is ``dataset/<Country Name>/<image>.jpg``. This module
turns that into PyTorch datasets while coping with two real-world issues in
this data:

* **Extreme imbalance** - some countries have thousands of images, others
  only one. We drop tiny classes and cap huge ones.
* **Stratified split** - every kept class appears in both train and val.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD, TrainConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def discover_samples(
    dataset_dir: Path,
    min_images_per_class: int,
    max_images_per_class: int,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[str]]:
    """Scan the dataset directory and return ``(samples, class_names)``.

    ``samples`` is a list of ``(image_path, class_index)`` tuples. Classes
    with fewer than ``min_images_per_class`` images are skipped; classes with
    more than ``max_images_per_class`` are randomly down-sampled.
    """
    rng = random.Random(seed)
    per_class: dict[str, list[Path]] = defaultdict(list)

    for class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        images = [p for p in class_dir.iterdir() if p.is_file() and _is_image(p)]
        if len(images) >= min_images_per_class:
            if len(images) > max_images_per_class:
                images = rng.sample(images, max_images_per_class)
            per_class[class_dir.name] = images

    class_names = sorted(per_class)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples: list[tuple[Path, int]] = []
    for name in class_names:
        idx = class_to_idx[name]
        samples.extend((path, idx) for path in per_class[name])

    return samples, class_names


def stratified_split(
    samples: list[tuple[Path, int]],
    val_split: float,
    num_classes: int,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Split per-class so every class is represented in both halves."""
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for item in samples:
        by_class[item[1]].append(item)

    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    for idx in range(num_classes):
        items = by_class[idx]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_split)) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_transforms(image_size: int, train: bool) -> Callable:
    """Return the torchvision transform pipeline.

    Note: GeoGuessr imagery is geographically meaningful, so we avoid
    horizontal flips that would mirror road signs / driving side and instead
    rely on mild colour/affine jitter for augmentation.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.RandomAffine(degrees=3, translate=(0.03, 0.03)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CountryDataset(Dataset):
    """A flat list of ``(image_path, class_idx)`` with a transform."""

    def __init__(self, samples: list[tuple[Path, int]], transform: Callable):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            # Corrupt/unreadable image: fall back to a black frame so a
            # single bad file never crashes a long training run.
            image = Image.new("RGB", (self.transform_size, self.transform_size))
        return self.transform(image), target

    @property
    def transform_size(self) -> int:  # used only by the fallback above
        return 224


def build_datasets(
    cfg: TrainConfig,
) -> tuple[CountryDataset, CountryDataset, list[str], list[int]]:
    """High-level entry point used by the training script.

    Returns ``(train_ds, val_ds, class_names, train_class_counts)``.
    """
    samples, class_names = discover_samples(
        cfg.dataset_dir,
        cfg.min_images_per_class,
        cfg.max_images_per_class,
        cfg.seed,
    )
    if not class_names:
        raise RuntimeError(
            f"No classes with >= {cfg.min_images_per_class} images found in "
            f"{cfg.dataset_dir}. Lower --min-images-per-class."
        )

    train_samples, val_samples = stratified_split(
        samples, cfg.val_split, len(class_names), cfg.seed
    )

    train_counts = [0] * len(class_names)
    for _, idx in train_samples:
        train_counts[idx] += 1

    train_ds = CountryDataset(train_samples, build_transforms(cfg.image_size, True))
    val_ds = CountryDataset(val_samples, build_transforms(cfg.image_size, False))
    return train_ds, val_ds, class_names, train_counts
