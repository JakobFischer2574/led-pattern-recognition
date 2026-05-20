from __future__ import annotations

import cv2
import numpy as np


def to_value_channel(roi_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2]


def create_green_led_mask(roi_bgr: np.ndarray, config: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Erstellt eine robuste Grünmaske für LED-Segmentierung.

    Returns:
        (mask, exg_score)
        - mask: Binäre uint8-Maske (0/255)
        - exg_score: Excess-Green-Score pro Pixel (float32)
    """
    seg_cfg = config.get("segmentation", {}) if isinstance(config, dict) else {}
    hsv_lower_cfg = seg_cfg.get("hsv_lower", {}) if isinstance(seg_cfg, dict) else {}
    hsv_upper_cfg = seg_cfg.get("hsv_upper", {}) if isinstance(seg_cfg, dict) else {}

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([
        int(hsv_lower_cfg.get("h", 35)),
        int(hsv_lower_cfg.get("s", 50)),
        int(hsv_lower_cfg.get("v", 80)),
    ], dtype=np.uint8)
    upper = np.array([
        int(hsv_upper_cfg.get("h", 95)),
        int(hsv_upper_cfg.get("s", 255)),
        int(hsv_upper_cfg.get("v", 255)),
    ], dtype=np.uint8)

    hsv_mask = cv2.inRange(hsv, lower, upper)

    b = roi_bgr[:, :, 0].astype(np.float32)
    g = roi_bgr[:, :, 1].astype(np.float32)
    r = roi_bgr[:, :, 2].astype(np.float32)
    exg_score = 2.0 * g - r - b

    min_exg = float(seg_cfg.get("min_excess_green", 30)) if isinstance(seg_cfg, dict) else 30.0
    exg_mask = np.where(exg_score >= min_exg, 255, 0).astype(np.uint8)

    combined = cv2.bitwise_and(hsv_mask, exg_mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned, exg_score
