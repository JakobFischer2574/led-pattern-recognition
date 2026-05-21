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


def create_led_masks(
    roi_bgr: np.ndarray,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Erstellt Grün-, Weißkern- und kombinierte LED-Maske inklusive Debug-Metriken."""
    green_mask, exg_score = create_green_led_mask(roi_bgr, config)
    seg_cfg = config.get("segmentation", {}) if isinstance(config, dict) else {}
    white_cfg = seg_cfg.get("white_core", {}) if isinstance(seg_cfg, dict) else {}
    adjacency_cfg = seg_cfg.get("white_core_adjacency", {}) if isinstance(seg_cfg, dict) else {}

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    max_saturation = int(white_cfg.get("max_saturation", 80)) if isinstance(white_cfg, dict) else 80
    min_value = int(white_cfg.get("min_value", 230)) if isinstance(white_cfg, dict) else 230
    white_core_mask = np.where((s < max_saturation) & (v > min_value), 255, 0).astype(np.uint8)

    dilate_kernel_size = int(adjacency_cfg.get("dilate_kernel_size", 3)) if isinstance(adjacency_cfg, dict) else 3
    dilate_iterations = int(adjacency_cfg.get("dilate_iterations", 1)) if isinstance(adjacency_cfg, dict) else 1
    dilate_kernel_size = max(1, dilate_kernel_size)
    if dilate_kernel_size % 2 == 0:
        dilate_kernel_size += 1
    adjacency_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)
    green_dilated = cv2.dilate(green_mask, adjacency_kernel, iterations=max(1, dilate_iterations))

    valid_white_core_mask = cv2.bitwise_and(white_core_mask, green_dilated)
    combined_led_mask = cv2.bitwise_or(green_mask, valid_white_core_mask)

    largest_area = 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((combined_led_mask > 0).astype(np.uint8), connectivity=8)
    if num_labels > 1:
        largest_area = float(np.max(stats[1:, cv2.CC_STAT_AREA]))

    debug_info = {
        "green_area": float(np.count_nonzero(green_mask)),
        "white_core_area": float(np.count_nonzero(white_core_mask)),
        "valid_white_core_area": float(np.count_nonzero(valid_white_core_mask)),
        "combined_led_area": float(np.count_nonzero(combined_led_mask)),
        "combined_largest_component_area": largest_area,
    }
    return green_mask, white_core_mask, valid_white_core_mask, combined_led_mask, {"exg": exg_score, **debug_info}
