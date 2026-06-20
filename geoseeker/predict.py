"""Inference helpers shared by the CLI, the explainer and the overlay app.

The :class:`Predictor` wraps a trained checkpoint and exposes a simple API:
turn a ``PIL.Image`` into a normalised tensor and into ranked country
predictions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .config import BEST_MODEL_PATH, IMAGENET_MEAN, IMAGENET_STD
from .model import load_checkpoint, resolve_device


@dataclass
class Prediction:
    country: str
    probability: float
    index: int


class Predictor:
    def __init__(self, checkpoint: Path = BEST_MODEL_PATH, device: str = "auto"):
        self.device = resolve_device(device)
        self.model, self.class_names, self.backbone, self.image_size = load_checkpoint(
            checkpoint, self.device
        )
        self._resize = transforms.Resize((self.image_size, self.image_size))
        self._normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self._to_tensor = transforms.ToTensor()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> normalised ``(1, 3, H, W)`` tensor on the model device."""
        image = image.convert("RGB")
        tensor = self._normalize(self._to_tensor(self._resize(image)))
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image, topk: int = 5) -> list[Prediction]:
        tensor = self.preprocess(image)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        k = min(topk, probs.numel())
        top = torch.topk(probs, k)
        return [
            Prediction(self.class_names[i], float(p), int(i))
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]


def main() -> None:
    p = argparse.ArgumentParser(description="Predict the country of an image.")
    p.add_argument("image", type=str, help="Path to an image file.")
    p.add_argument("--checkpoint", type=str, default=str(BEST_MODEL_PATH))
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    predictor = Predictor(Path(args.checkpoint), args.device)
    image = Image.open(args.image)
    print(f"\nPredictions for {args.image}:")
    for rank, pred in enumerate(predictor.predict(image, args.topk), start=1):
        bar = "#" * int(pred.probability * 30)
        print(f"  {rank}. {pred.country:<28} {pred.probability*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
