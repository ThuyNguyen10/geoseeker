# GeoSeeker

Guess the **country** from a street-view image with a PyTorch classifier, see
**why** the model guessed it via Captum Grad-CAM, and run a live **overlay**
while you play GeoGuessr.

```
geoseeker/
  config.py     # paths + hyper-parameters
  data.py       # dataset discovery, balancing, stratified split, transforms
  model.py      # transfer-learning backbones + checkpoint load/save
  train.py      # training loop (CPU-friendly)
  predict.py    # Predictor class + CLI
  explain.py    # Captum Grad-CAM heatmap overlay
  overlay.py    # always-on-top GeoGuessr overlay app
```

## 1. Install

```bash
pip install -r requirements.txt
```

CPU-only is fine. If you have an NVIDIA GPU, install the CUDA build of torch
from https://pytorch.org for much faster training.

## 2. Train

The dataset lives in `dataset/<Country>/<image>.jpg`. It is heavily imbalanced
(United States has ~12k images, some countries only one), so training:

- **drops** classes with fewer than `--min-images-per-class` images (default 20),
- **caps** huge classes at `--max-images-per-class` (default 1500),
- **balances** batches with a weighted sampler.

Fast first model (train only the head — minutes on CPU):

```bash
python -m geoseeker.train --freeze-backbone --epochs 5
```

Higher accuracy (fine-tune the whole backbone — slow on CPU, fast on GPU):

```bash
python -m geoseeker.train --backbone resnet18 --epochs 15
```

The best checkpoint is written to `checkpoints/geoseeker_best.pt` and bundles
the class names, backbone and image size, so every other tool is self-contained.

Key flags: `--backbone {resnet18,resnet50,mobilenet_v3_large}`,
`--image-size`, `--batch-size`, `--lr`, `--device {auto,cpu,cuda}`.

## 3. Predict a single image

```bash
python -m geoseeker.predict path/to/street.jpg --topk 5
```

## 4. Explain (Captum Grad-CAM)

See which regions drove the guess:

```bash
python -m geoseeker.explain path/to/street.jpg --out explanation.png
```

Produces a heatmap overlay (red = most influential for the predicted country).

## 5. Live overlay for GeoGuessr

```bash
python -m geoseeker.overlay                      # capture primary monitor
python -m geoseeker.overlay --region 100,80,1280,720
python -m geoseeker.overlay --monitor 2
```

A small, draggable, always-on-top panel appears. Global hotkeys:

| Hotkey            | Action                                          |
| ----------------- | ----------------------------------------------- |
| `Ctrl+Shift+G`    | Grab screen and show top-5 guesses              |
| `Ctrl+Shift+E`    | Grab + open the Grad-CAM explanation            |
| `Ctrl+Shift+L`    | Toggle live mode (auto-guess every few seconds) |
| `Ctrl+Shift+X`    | Toggle Grad-CAM in live mode                     |
| `Ctrl+Shift+Q`    | Quit                                            |

Start live mode with Grad-CAM already on via `--live-explain`. In live mode the
explanation window refreshes automatically each tick (slightly slower interval
since Grad-CAM adds compute on CPU).

Notes:
- Global hotkeys and screen capture work best under **X11**. On Wayland,
  capture can be restricted — if hotkeys don't fire, focus the overlay window
  and use the same keys, or run the session under Xorg.
- Use `--region` to capture just the GeoGuessr panorama (exclude the minimap/UI)
  for cleaner predictions.

## Accuracy expectations

With 124 candidate countries and noisy crowd-collected imagery, treat top-5 as
the useful signal. Fine-tuning the backbone and using a larger `--image-size`
improves results at the cost of speed.
