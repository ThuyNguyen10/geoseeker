"""Live GeoGuessr overlay.

A small always-on-top, semi-transparent window that sits over your GeoGuessr
session. Press a hotkey to grab the current screen, run the classifier and show
the top country guesses - plus, on demand, a Captum Grad-CAM heatmap revealing
which parts of the street view drove the guess.

Hotkeys (global, work while GeoGuessr has focus):

* ``Ctrl+Shift+G`` - grab screen and predict
* ``Ctrl+Shift+E`` - grab, predict and open the Grad-CAM explanation window
* ``Ctrl+Shift+L`` - toggle live mode (auto-predict every few seconds)
* ``Ctrl+Shift+Q`` - quit

Run with::

    python -m geoseeker.overlay                 # capture primary monitor
    python -m geoseeker.overlay --region 100,80,1280,720
    python -m geoseeker.overlay --monitor 2     # capture monitor #2

Notes:
* Global hotkeys and screen capture work best under X11. On Wayland, capture
  may be restricted - run the session under Xorg if hotkeys don't fire.
"""
from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageTk

from .config import BEST_MODEL_PATH
from .explain import GradCamExplainer
from .predict import Predictor


@dataclass
class CaptureSpec:
    """What region of the screen to grab. ``None`` region => full monitor."""

    monitor: int = 1
    region: tuple[int, int, int, int] | None = None  # (left, top, width, height)


class GeoSeekerOverlay:
    POLL_MS = 100

    def __init__(
        self,
        predictor: Predictor,
        capture: CaptureSpec,
        alpha: float,
        live_explain: bool = False,
    ):
        self.predictor = predictor
        self.explainer = GradCamExplainer(predictor)
        self.capture = capture

        self.results: "queue.Queue" = queue.Queue()
        self.live_mode = False
        # When True, live mode also renders the Grad-CAM explanation each tick.
        self.live_explain = live_explain
        self._explain_window: tk.Toplevel | None = None
        self._explain_photo: ImageTk.PhotoImage | None = None
        self._busy = False

        self._build_ui(alpha)
        self._bind_keys()
        self.root.after(self.POLL_MS, self._drain_results)

    # ------------------------------------------------------------------ UI
    def _build_ui(self, alpha: float) -> None:
        self.root = tk.Tk()
        self.root.title("GeoSeeker")
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", alpha)
        except tk.TclError:
            pass  # alpha unsupported on some platforms
        self.root.configure(bg="#101418")
        self.root.geometry("+40+40")
        self.root.overrideredirect(False)

        header = tk.Frame(self.root, bg="#101418")
        header.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(
            header,
            text="GeoSeeker",
            fg="#7ee787",
            bg="#101418",
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")
        self.status = tk.Label(
            header, text="ready", fg="#8b949e", bg="#101418", font=("Segoe UI", 9)
        )
        self.status.pack(side="right")

        self.body = tk.Frame(self.root, bg="#101418")
        self.body.pack(fill="both", expand=True, padx=10, pady=(2, 6))
        self.pred_label = tk.Label(
            self.body,
            text="Press Ctrl+Shift+G to guess",
            fg="#c9d1d9",
            bg="#101418",
            justify="left",
            anchor="w",
            font=("Consolas", 11),
        )
        self.pred_label.pack(fill="both", expand=True)

        footer = tk.Frame(self.root, bg="#0b0e11")
        footer.pack(fill="x")
        tk.Label(
            footer,
            text="G guess   E explain   L live   X live+cam   Q quit",
            fg="#6e7681",
            bg="#0b0e11",
            font=("Segoe UI", 8),
        ).pack(padx=10, pady=3)

        # Allow dragging the window by its body.
        for widget in (header, self.body, self.pred_label):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._on_drag)

    def _start_drag(self, event: tk.Event) -> None:
        self._drag_x, self._drag_y = event.x, event.y

    def _on_drag(self, event: tk.Event) -> None:
        x = self.root.winfo_x() + event.x - self._drag_x
        y = self.root.winfo_y() + event.y - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    def _bind_keys(self) -> None:
        # In-window bindings (focus on the overlay).
        self.root.bind("<Control-Shift-G>", lambda e: self.trigger(explain=False))
        self.root.bind("<Control-Shift-g>", lambda e: self.trigger(explain=False))
        self.root.bind("<Control-Shift-E>", lambda e: self.trigger(explain=True))
        self.root.bind("<Control-Shift-e>", lambda e: self.trigger(explain=True))
        self.root.bind("<Control-Shift-L>", lambda e: self.toggle_live())
        self.root.bind("<Control-Shift-l>", lambda e: self.toggle_live())
        self.root.bind("<Control-Shift-X>", lambda e: self.toggle_live_explain())
        self.root.bind("<Control-Shift-x>", lambda e: self.toggle_live_explain())
        self.root.bind("<Control-Shift-Q>", lambda e: self.quit())
        self.root.bind("<Control-Shift-q>", lambda e: self.quit())

        # Global hotkeys (work while another app is focused).
        self._start_global_hotkeys()

    def _start_global_hotkeys(self) -> None:
        try:
            from pynput import keyboard
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[overlay] global hotkeys disabled ({exc}). Use in-window keys.")
            return

        def on(fn):
            # pynput callbacks run on a listener thread; marshal to Tk thread.
            return lambda: self.root.after(0, fn)

        hotkeys = keyboard.GlobalHotKeys(
            {
                "<ctrl>+<shift>+g": on(lambda: self.trigger(explain=False)),
                "<ctrl>+<shift>+e": on(lambda: self.trigger(explain=True)),
                "<ctrl>+<shift>+l": on(self.toggle_live),
                "<ctrl>+<shift>+x": on(self.toggle_live_explain),
                "<ctrl>+<shift>+q": on(self.quit),
            }
        )
        hotkeys.daemon = True
        hotkeys.start()

    # ------------------------------------------------------------- capture
    def _grab(self) -> Image.Image:
        """Capture the configured screen region. Created per-call: the mss
        instance must live on the thread that uses it."""
        import mss

        with mss.mss() as sct:
            if self.capture.region is not None:
                left, top, width, height = self.capture.region
                bbox = {"left": left, "top": top, "width": width, "height": height}
            else:
                bbox = sct.monitors[self.capture.monitor]
            shot = sct.grab(bbox)
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    # ------------------------------------------------------------- actions
    def trigger(self, explain: bool) -> None:
        if self._busy:
            return
        self._busy = True
        self.status.config(text="thinking...", fg="#d29922")
        threading.Thread(target=self._worker, args=(explain,), daemon=True).start()

    def _worker(self, explain: bool) -> None:
        try:
            frame = self._grab()
            preds = self.predictor.predict(frame, topk=5)
            overlay_img = None
            if explain:
                overlay_img, _ = self.explainer.overlay(frame, preds[0].index)
            self.results.put(("ok", preds, overlay_img))
        except Exception as exc:  # surface errors to the UI
            self.results.put(("error", str(exc), None))
        finally:
            self._busy = False

    def _drain_results(self) -> None:
        try:
            while True:
                kind, payload, overlay_img = self.results.get_nowait()
                if kind == "error":
                    self.status.config(text="error", fg="#f85149")
                    self.pred_label.config(text=f"Error:\n{payload}")
                else:
                    self._show_predictions(payload)
                    if overlay_img is not None:
                        self._show_explanation(overlay_img)
                    self.status.config(text="ready", fg="#8b949e")
        except queue.Empty:
            pass
        finally:
            self.root.after(self.POLL_MS, self._drain_results)

    def _show_predictions(self, preds) -> None:
        lines = []
        for rank, pred in enumerate(preds, start=1):
            bar = "#" * int(pred.probability * 18)
            lines.append(
                f"{rank}. {pred.country[:20]:<20} {pred.probability*100:5.1f}% {bar}"
            )
        self.pred_label.config(text="\n".join(lines))

    def _show_explanation(self, image: Image.Image) -> None:
        # Scale to a comfortable preview width.
        max_w = 640
        if image.width > max_w:
            ratio = max_w / image.width
            image = image.resize((max_w, int(image.height * ratio)), Image.BILINEAR)

        if self._explain_window is None or not self._explain_window.winfo_exists():
            self._explain_window = tk.Toplevel(self.root)
            self._explain_window.title("GeoSeeker - where the model looked")
            self._explain_window.attributes("-topmost", True)
            self._explain_label = tk.Label(self._explain_window, bg="#000000")
            self._explain_label.pack()

        self._explain_photo = ImageTk.PhotoImage(image)
        self._explain_label.config(image=self._explain_photo)

    def toggle_live(self) -> None:
        self.live_mode = not self.live_mode
        self._update_live_status()
        if self.live_mode:
            self._live_tick()

    def toggle_live_explain(self) -> None:
        """Toggle whether live mode also renders the Grad-CAM explanation."""
        self.live_explain = not self.live_explain
        self._update_live_status()

    def _update_live_status(self) -> None:
        if self.live_mode:
            text = "live ON + cam" if self.live_explain else "live ON"
            self.status.config(text=text, fg="#7ee787")
        else:
            text = "live+cam ready" if self.live_explain else "live OFF"
            self.status.config(text=text, fg="#8b949e")

    def _live_tick(self) -> None:
        if not self.live_mode:
            return
        self.trigger(explain=self.live_explain)
        # Grad-CAM costs extra compute, so poll a little slower when it's on.
        interval = 4500 if self.live_explain else 3000
        self.root.after(interval, self._live_tick)

    def quit(self) -> None:
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _parse_region(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [int(x) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--region must be 'left,top,width,height'")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    p = argparse.ArgumentParser(description="Live GeoGuessr overlay.")
    p.add_argument("--checkpoint", type=str, default=str(BEST_MODEL_PATH))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--monitor", type=int, default=1, help="mss monitor index (1=primary).")
    p.add_argument(
        "--region",
        type=str,
        default=None,
        help="Capture region 'left,top,width,height'. Overrides --monitor.",
    )
    p.add_argument("--alpha", type=float, default=0.9, help="Overlay window opacity.")
    p.add_argument(
        "--live-explain",
        action="store_true",
        help="Show the Grad-CAM explanation in live mode (Ctrl+Shift+X toggles it).",
    )
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint}\n"
            "Train a model first:  python -m geoseeker.train --freeze-backbone --epochs 5"
        )

    predictor = Predictor(checkpoint, args.device)
    capture = CaptureSpec(monitor=args.monitor, region=_parse_region(args.region))
    print(f"Loaded {len(predictor.class_names)} countries. Overlay starting...")
    GeoSeekerOverlay(predictor, capture, args.alpha, args.live_explain).run()


if __name__ == "__main__":
    main()
