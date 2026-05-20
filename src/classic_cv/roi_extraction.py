from __future__ import annotations

import numpy as np


def extract_rois(frame: np.ndarray, led_layout: dict[str, dict[str, int]]) -> dict[str, np.ndarray]:
    rois: dict[str, np.ndarray] = {}
    for name, roi in led_layout.items():
        x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
        rois[name] = frame[y : y + h, x : x + w]
    return rois
