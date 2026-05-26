from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.classic_cv.feature_extraction import compute_brightness_features, compute_green_features
from src.classic_cv.led_state_classifier import classify_led_state
from src.classic_cv.locators import FixedROILocator, SlotBasedLEDLocator, TrackingLEDLocator
from src.classic_cv.preprocessing import preprocess_frame
from src.classic_cv.segmentation import create_led_masks, to_value_channel
from src.detectors.base_detector import BaseDetector, DetectionResult
from src.utils.image_debug import draw_led_debug_overlay, save_debug_image


class ClassicCVDetector(BaseDetector):
    def __init__(self, config: dict[str, Any], led_layout: dict[str, dict[str, int]]) -> None:
        self.config = config
        self.led_layout = led_layout
        locator_cfg = config.get("locator", {})
        locator_type = str(locator_cfg.get("type", "fixed_roi"))
        if locator_type == "fixed_roi":
            self.locator = FixedROILocator(led_layout)
        elif locator_type == "slot_based":
            slot_locator = SlotBasedLEDLocator(locator_cfg)
            tracking_cfg = locator_cfg.get("tracking", {})
            self.locator = TrackingLEDLocator(
                slot_locator,
                enabled=bool(tracking_cfg.get("enabled", True)),
                max_tracking_fallback_frames=int(tracking_cfg.get("max_tracking_fallback_frames", 5)),
                fallback_confidence=float(tracking_cfg.get("fallback_confidence", 0.5)),
            )
        else:
            raise ValueError(f"Ungültiger locator.type: {locator_type}. Erlaubt: fixed_roi, slot_based")
        self.locator_type = locator_type

    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        prep_cfg = self.config.get("preprocessing", {})
        cls_cfg = self.config.get("classification", {})
        seg_cfg = self.config.get("segmentation", {})
        use_combined_led_mask = bool(seg_cfg.get("use_combined_led_mask_for_classification", True))
        processed = preprocess_frame(frame, prep_cfg.get("resize_width"), int(prep_cfg.get("blur_kernel_size", 5)))

        locator_result = self.locator.locate(processed)
        if locator_result.status == "failed" or len(locator_result.regions) != 5:
            dt_ms = (time.perf_counter() - start) * 1000
            return DetectionResult(
                led_state=[-1, -1, -1, -1, -1],
                confidences=[0.0, 0.0, 0.0, 0.0, 0.0],
                processing_time_ms=dt_ms,
                locator_status="failed",
                locator_confidence=0.0,
                debug_info={
                    "metrics": [],
                    "processed_frame": processed,
                    "original_frame": frame,
                    "locator": locator_result.debug_info,
                    "locator_status": "failed",
                },
            )

        led_state: list[int] = []
        confidences: list[float] = []
        metrics_debug: list[dict[str, Any]] = []

        sorted_regions = sorted(locator_result.regions, key=lambda r: r.led_id)
        for region in sorted_regions:
            x, y, w, h = region.x, region.y, region.width, region.height
            roi_img = processed[y : y + h, x : x + w]
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
            metrics_debug.append(
                {
                    "led_id": region.led_id,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
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
                    "name": region.led_id,
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
            locator_status=locator_result.status,
            locator_confidence=float(locator_result.confidence),
            debug_info={
                "metrics": metrics_debug,
                "processed_frame": processed,
                "original_frame": frame,
                "locator": locator_result.debug_info,
                "locator_status": locator_result.status,
                "locator_confidence": float(locator_result.confidence),
                "locator_type": self.locator_type,
            },
        )

    def save_debug(self, output_path: str | Path, result: DetectionResult) -> None:
        rois = {m["led_id"]: {"x": m["x"], "y": m["y"], "width": m["width"], "height": m["height"]} for m in result.debug_info.get("metrics", [])}
        overlay = draw_led_debug_overlay(result.debug_info["processed_frame"], result.debug_info.get("metrics", []), rois)
        save_debug_image(output_path, overlay)
