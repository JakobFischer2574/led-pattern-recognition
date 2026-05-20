from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def draw_led_debug_overlay(frame: np.ndarray, led_metrics: list[dict[str, Any]], rois: dict[str, dict[str, int]]) -> np.ndarray:
    canvas = frame.copy()
    for idx, (name, roi) in enumerate(rois.items()):
        x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
        metrics = led_metrics[idx]
        state = metrics["state"]
        color = (0, 255, 0) if state == 1 else (0, 0, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        text = f"{name}:{state} m={metrics['mean']:.1f} mx={metrics['max']:.1f} r={metrics['ratio']:.2f}"
        cv2.putText(canvas, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas


def save_debug_image(path: str | Path, image: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise IOError(f"Debug-Bild konnte nicht gespeichert werden: {output}")
