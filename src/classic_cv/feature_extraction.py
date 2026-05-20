from __future__ import annotations

import numpy as np


def compute_brightness_features(value_channel: np.ndarray, bright_threshold: int) -> dict[str, float]:
    mean_brightness = float(np.mean(value_channel))
    max_brightness = float(np.max(value_channel))
    bright_ratio = float(np.mean(value_channel >= bright_threshold))
    return {
        "mean_brightness": mean_brightness,
        "max_brightness": max_brightness,
        "bright_pixel_ratio": bright_ratio,
    }
