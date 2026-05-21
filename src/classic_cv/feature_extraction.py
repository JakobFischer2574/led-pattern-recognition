from __future__ import annotations

import cv2
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


def compute_green_features(
    green_mask: np.ndarray,
    exg_score: np.ndarray,
    classification_mask: np.ndarray | None = None,
    segmentation_debug: dict[str, float] | None = None,
) -> dict[str, float]:
    used_mask = classification_mask if classification_mask is not None else green_mask
    area = float(used_mask.shape[0] * used_mask.shape[1])
    green_pixels = float(np.count_nonzero(used_mask))
    green_ratio = (green_pixels / area) if area > 0 else 0.0

    if green_pixels > 0:
        exg_in_mask = exg_score[used_mask > 0]
        mean_green_score = float(np.mean(exg_in_mask))
        max_green_score = float(np.max(exg_in_mask))
    else:
        mean_green_score = 0.0
        max_green_score = 0.0

    largest_area = 0.0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((used_mask > 0).astype(np.uint8), connectivity=8)
    if num_labels > 1:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        if component_areas.size > 0:
            largest_area = float(np.max(component_areas))

    features = {
        "green_area": green_pixels,
        "green_pixel_ratio": green_ratio,
        "mean_green_score": mean_green_score,
        "max_green_score": max_green_score,
        "largest_green_component_area": largest_area,
    }
    if segmentation_debug:
        features.update(segmentation_debug)
    return features
