"""Explainability with Captum.

We use Captum's :class:`LayerGradCam` on the last convolutional block to build
a heatmap of *where* the model looks when it picks a country, then alpha-blend
it over the original frame. Grad-CAM is cheap enough to run on CPU in close to
real time, which matters for the live overlay.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import LayerAttribution, LayerGradCam
from PIL import Image

from .config import BEST_MODEL_PATH
from .model import gradcam_target_layer
from .predict import Predictor


def _colorize(heat: np.ndarray) -> np.ndarray:
    """Map a [0,1] heatmap to an RGB array using a blue->red ramp.

    Implemented without matplotlib so the overlay has one less dependency.
    """
    heat = np.clip(heat, 0.0, 1.0)
    # Simple 3-stop colormap: blue (cold) -> green -> red (hot).
    r = np.clip(1.5 * heat - 0.5, 0, 1)
    g = np.clip(1.0 - np.abs(heat - 0.5) * 2.0, 0, 1)
    b = np.clip(1.0 - 1.5 * heat, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


class GradCamExplainer:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor
        target_layer = gradcam_target_layer(predictor.model, predictor.backbone)
        self.gradcam = LayerGradCam(predictor.model, target_layer)

    def attribution(self, image: Image.Image, target_index: int) -> np.ndarray:
        """Return a ``(H, W)`` heatmap in [0,1] at the model's input size."""
        tensor = self.predictor.preprocess(image)
        tensor.requires_grad_(True)

        attr = self.gradcam.attribute(tensor, target=target_index, relu_attributions=True)
        size = self.predictor.image_size
        attr = LayerAttribution.interpolate(attr, (size, size), interpolate_mode="bilinear")
        heat = attr.squeeze().detach().cpu().numpy()

        heat = np.maximum(heat, 0)
        if heat.max() > 0:
            heat = heat / heat.max()
        return heat

    def overlay(
        self,
        image: Image.Image,
        target_index: int | None = None,
        alpha: float = 0.45,
    ) -> tuple[Image.Image, int]:
        """Return ``(blended_image, target_index)``.

        If ``target_index`` is ``None`` the model's top prediction is used.
        """
        if target_index is None:
            target_index = self.predictor.predict(image, topk=1)[0].index

        heat = self.attribution(image, target_index)
        base = image.convert("RGB")
        heat_img = Image.fromarray(_colorize(heat)).resize(base.size, Image.BILINEAR)

        base_arr = np.asarray(base).astype(np.float32)
        heat_arr = np.asarray(heat_img).astype(np.float32)
        # Weight the blend by heat intensity so cold regions stay clear.
        weight = (heat * alpha)
        weight_img = np.asarray(
            Image.fromarray((weight * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR)
        ).astype(np.float32)[..., None] / 255.0

        blended = base_arr * (1 - weight_img) + heat_arr * weight_img
        return Image.fromarray(blended.clip(0, 255).astype(np.uint8)), target_index


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a Grad-CAM explanation overlay.")
    p.add_argument("image", type=str, help="Path to an image file.")
    p.add_argument("--checkpoint", type=str, default=str(BEST_MODEL_PATH))
    p.add_argument("--out", type=str, default="explanation.png")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    predictor = Predictor(Path(args.checkpoint), args.device)
    explainer = GradCamExplainer(predictor)
    image = Image.open(args.image)

    preds = predictor.predict(image, topk=5)
    print("Top predictions:")
    for rank, pred in enumerate(preds, start=1):
        print(f"  {rank}. {pred.country:<28} {pred.probability*100:5.1f}%")

    overlay, idx = explainer.overlay(image, preds[0].index, args.alpha)
    overlay.save(args.out)
    print(f"\nSaved explanation for '{predictor.class_names[idx]}' -> {args.out}")


if __name__ == "__main__":
    main()
