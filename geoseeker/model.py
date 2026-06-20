"""Model construction and checkpoint (de)serialisation.

We use transfer learning from torchvision backbones. The classifier head is
swapped for one that outputs ``num_classes`` logits. Checkpoints bundle the
weights together with everything needed to reproduce inference (class names,
backbone name, image size) so the predictor and overlay are self-contained.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_BACKBONES = ("resnet18", "resnet50", "mobilenet_v3_large")


def _make_backbone(backbone: str, num_classes: int, weights):
    if backbone == "resnet18":
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif backbone == "resnet50":
        net = models.resnet50(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif backbone == "mobilenet_v3_large":
        net = models.mobilenet_v3_large(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose from {SUPPORTED_BACKBONES}."
        )
    return net


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a torchvision backbone with a fresh classification head.

    If ``pretrained`` is requested but the weights cannot be downloaded (e.g.
    an offline machine or a TLS-intercepting corporate proxy), we fall back to
    randomly initialised weights instead of crashing, and print guidance.
    """
    if not pretrained:
        return _make_backbone(backbone, num_classes, weights=None)

    try:
        return _make_backbone(backbone, num_classes, weights="DEFAULT")
    except Exception as exc:  # download / SSL failures land here
        print(
            "[model] Could not download pretrained weights "
            f"({type(exc).__name__}: {exc}).\n"
            "[model] Falling back to RANDOM initialisation. To use pretrained "
            "weights offline, pre-download the .pth into "
            "~/.cache/torch/hub/checkpoints/ or fix your CA bundle "
            "(set SSL_CERT_FILE to a bundle that trusts your proxy)."
        )
        return _make_backbone(backbone, num_classes, weights=None)


def freeze_backbone(model: nn.Module, backbone: str) -> None:
    """Freeze every parameter except the final classification head."""
    for param in model.parameters():
        param.requires_grad = False

    if backbone in ("resnet18", "resnet50"):
        head = model.fc
    else:  # mobilenet_v3_large
        head = model.classifier[-1]
    for param in head.parameters():
        param.requires_grad = True


def gradcam_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    """Return the last convolutional layer, used as the Grad-CAM target."""
    if backbone in ("resnet18", "resnet50"):
        return model.layer4[-1]
    return model.features[-1]  # mobilenet_v3_large


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    backbone: str,
    class_names: list[str],
    image_size: int,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "backbone": backbone,
        "class_names": class_names,
        "image_size": image_size,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[nn.Module, list[str], str, int]:
    """Rebuild a ready-to-eval model from a checkpoint file."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    backbone = ckpt["backbone"]
    image_size = ckpt.get("image_size", 224)

    model = build_model(backbone, len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, class_names, backbone, image_size
