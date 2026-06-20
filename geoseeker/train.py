"""Training entry point.

Run from the project root, e.g.::

    python -m geoseeker.train --epochs 15 --backbone resnet18

The script is CPU-friendly by default. To get a fast first model, freeze the
backbone and train only the head::

    python -m geoseeker.train --freeze-backbone --epochs 5
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import TrainConfig
from .data import build_datasets
from .model import (
    build_model,
    freeze_backbone,
    resolve_device,
    save_checkpoint,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_sampler(train_counts: list[int], targets: list[int]) -> WeightedRandomSampler:
    """Balance batches so rare countries are seen as often as common ones."""
    class_weights = [1.0 / max(1, c) for c in train_counts]
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    top5_correct = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        k = min(5, logits.size(1))
        top5 = logits.topk(k, dim=1).indices
        top5_correct += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()

    return correct / max(1, total), top5_correct / max(1, total)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running = 0.0
    seen = 0
    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running += loss.item() * targets.size(0)
        seen += targets.size(0)
        pbar.set_postfix(loss=f"{running / max(1, seen):.3f}")
    return running / max(1, seen)


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description="Train the GeoSeeker country classifier.")
    p.add_argument("--dataset-dir", type=str, default=str(cfg.dataset_dir))
    p.add_argument("--image-size", type=int, default=cfg.image_size)
    p.add_argument("--min-images-per-class", type=int, default=cfg.min_images_per_class)
    p.add_argument("--max-images-per-class", type=int, default=cfg.max_images_per_class)
    p.add_argument("--val-split", type=float, default=cfg.val_split)
    p.add_argument("--backbone", type=str, default=cfg.backbone)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--device", type=str, default=cfg.device)
    p.add_argument("--output", type=str, default=str(cfg.output))
    args = p.parse_args()

    from pathlib import Path

    cfg.dataset_dir = Path(args.dataset_dir)
    cfg.image_size = args.image_size
    cfg.min_images_per_class = args.min_images_per_class
    cfg.max_images_per_class = args.max_images_per_class
    cfg.val_split = args.val_split
    cfg.backbone = args.backbone
    cfg.pretrained = not args.no_pretrained
    cfg.freeze_backbone = args.freeze_backbone
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.output = Path(args.output)
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Device: {device}")

    train_ds, val_ds, class_names, train_counts = build_datasets(cfg)
    print(
        f"Classes kept: {len(class_names)} | "
        f"train: {len(train_ds)} | val: {len(val_ds)}"
    )

    targets = [t for _, t in train_ds.samples]
    sampler = make_sampler(train_counts, targets)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg.backbone, len(class_names), cfg.pretrained).to(device)
    if cfg.freeze_backbone:
        freeze_backbone(model, cfg.backbone)
        print("Backbone frozen - training classification head only.")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        top1, top5 = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0
        print(
            f"epoch {epoch:3d} | loss {loss:.3f} | "
            f"val top1 {top1*100:5.2f}% | val top5 {top5*100:5.2f}% | {dt:.0f}s"
        )

        if top1 >= best_acc:
            best_acc = top1
            save_checkpoint(
                cfg.output,
                model,
                cfg.backbone,
                class_names,
                cfg.image_size,
                extra={"val_top1": top1, "val_top5": top5, "epoch": epoch},
            )
            print(f"  saved new best -> {cfg.output} (top1 {top1*100:.2f}%)")

    print(f"Done. Best val top1: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
