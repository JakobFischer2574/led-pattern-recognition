from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.classic_cv.feature_extraction import compute_brightness_features, compute_green_features
from src.classic_cv.led_state_classifier import classify_led_state
from src.classic_cv.preprocessing import preprocess_frame
from src.classic_cv.roi_extraction import extract_rois
from src.classic_cv.segmentation import create_led_masks, to_value_channel
from src.detectors.base_detector import BaseDetector, DetectionResult
from src.utils.image_debug import draw_led_debug_overlay, save_debug_image


class ClassicCVDetector(BaseDetector):
    def __init__(self, config: dict[str, Any], led_layout: dict[str, dict[str, int]]) -> None:
        self.config = config
        self.led_layout = led_layout

    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        prep_cfg = self.config.get("preprocessing", {})
        cls_cfg = self.config.get("classification", {})
        seg_cfg = self.config.get("segmentation", {})
        use_combined_led_mask = bool(seg_cfg.get("use_combined_led_mask_for_classification", True))
        processed = preprocess_frame(frame, prep_cfg.get("resize_width"), int(prep_cfg.get("blur_kernel_size", 5)))

        rois = extract_rois(processed, self.led_layout)
        led_state: list[int] = []
        confidences: list[float] = []
        metrics_debug: list[dict[str, Any]] = []

        for led_name in sorted(rois.keys()):
            roi_img = rois[led_name]
            value = to_value_channel(roi_img)
            brightness_features = compute_brightness_features(value, int(cls_cfg.get("brightness_threshold", 200)))
            green_mask, white_core_mask, valid_white_core_mask, combined_led_mask, seg_debug = create_led_masks(roi_img, self.config)
            classification_mask = combined_led_mask if use_combined_led_mask else green_mask
            green_features = compute_green_features(
                green_mask,
                seg_debug["exg"],
                classification_mask=classification_mask,
                segmentation_debug={k: float(v) for k, v in seg_debug.items() if k != "exg"},
            )
            features = {**brightness_features, **green_features}
            state, conf = classify_led_state(features, cls_cfg)
            led_state.append(state)
            confidences.append(conf)
            roi = self.led_layout[led_name]
            metrics_debug.append(
                {
                    "led_id": led_name,
                    "x": int(roi["x"]),
                    "y": int(roi["y"]),
                    "width": int(roi["width"]),
                    "height": int(roi["height"]),
                    "state": state,
                    "mean_brightness": float(features["mean_brightness"]),
                    "max_brightness": float(features["max_brightness"]),
                    "bright_pixel_ratio": float(features["bright_pixel_ratio"]),
                    "green_area": float(features["green_area"]),
                    "green_pixel_ratio": float(features["green_pixel_ratio"]),
                    "mean_green_score": float(features["mean_green_score"]),
                    "max_green_score": float(features["max_green_score"]),
                    "largest_green_component_area": float(features["largest_green_component_area"]),
                    "white_core_area": float(features["white_core_area"]),
                    "valid_white_core_area": float(features["valid_white_core_area"]),
                    "combined_led_area": float(features["combined_led_area"]),
                    "combined_largest_component_area": float(features["combined_largest_component_area"]),
                    "green_mask": green_mask,
                    "white_core_mask": white_core_mask,
                    "combined_led_mask": combined_led_mask,
                    "classification_mask_type": "combined" if use_combined_led_mask else "green",
                    "confidence": float(conf),
                    # Backward-compatible keys for existing overlay path.
                    "name": led_name,
                    "mean": float(features["mean_brightness"]),
                    "max": float(features["max_brightness"]),
                    "ratio": float(features["bright_pixel_ratio"]),
                }
            )

        dt_ms = (time.perf_counter() - start) * 1000
        return DetectionResult(
            led_state=led_state,
            confidences=confidences,
            processing_time_ms=dt_ms,
            debug_info={"metrics": metrics_debug, "processed_frame": processed, "original_frame": frame},
        )

    def save_debug(self, output_path: str | Path, result: DetectionResult) -> None:
        overlay = draw_led_debug_overlay(result.debug_info["processed_frame"], result.debug_info["metrics"], self.led_layout)
        save_debug_image(output_path, overlay)
