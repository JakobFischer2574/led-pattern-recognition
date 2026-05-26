from __future__ import annotations

from itertools import combinations
from typing import Any

import cv2
import numpy as np

from src.classic_cv.locators.base_led_locator import BaseLEDLocator, LEDRegion, LocatorResult


class SlotBasedLEDLocator(BaseLEDLocator):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.search_region = config.get("search_region", {})
        self.slot_detection = config.get("slot_detection", {})
        self.geometry = config.get("geometry", {})
        self.roi_cfg = config.get("roi", {})
        self.expected_led_count = int(self.geometry.get("expected_led_count", 5))
        if self.expected_led_count != 5:
            raise ValueError("SlotBasedLEDLocator erwartet expected_led_count=5 für dieses Routermodell.")

    def _search_bounds(self, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        x0 = int(w * float(self.search_region.get("x_min_ratio", 0.05)))
        x1 = int(w * float(self.search_region.get("x_max_ratio", 0.95)))
        y0 = int(h * float(self.search_region.get("y_min_ratio", 0.35)))
        y1 = int(h * float(self.search_region.get("y_max_ratio", 0.90)))
        x0, x1 = max(0, min(x0, w - 1)), max(1, min(x1, w))
        y0, y1 = max(0, min(y0, h - 1)), max(1, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Ungültige locator.search_region Konfiguration.")
        return x0, y0, x1, y1

    def _score_group(self, group: list[tuple[int, int, int, int]]) -> float:
        centers = [x + w / 2.0 for x, _, w, _ in group]
        y_centers = [y + h / 2.0 for _, y, _, h in group]
        spacings = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        mean_spacing = max(1.0, float(np.mean(spacings)))
        spacing_dev = float(np.std(spacings) / mean_spacing)
        y_dev = float(np.std(y_centers))
        return spacing_dev + (y_dev / 100.0)

    def _select_plausible_five(self, candidates: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        max_y_dev = float(self.geometry.get("max_y_deviation_px", 40))
        max_spacing_dev_ratio = float(self.geometry.get("max_spacing_deviation_ratio", 0.45))
        if len(candidates) < self.expected_led_count:
            return []

        best_group: list[tuple[int, int, int, int]] = []
        best_score = float("inf")
        for combo in combinations(candidates, self.expected_led_count):
            group = sorted(combo, key=lambda c: c[0])
            ys = np.array([y + h / 2.0 for _, y, _, h in group], dtype=np.float32)
            if float(ys.max() - ys.min()) > max_y_dev:
                continue
            xs = np.array([x + w / 2.0 for x, _, w, _ in group], dtype=np.float32)
            spacings = np.diff(xs)
            if np.any(spacings <= 0):
                continue
            mean_spacing = float(np.mean(spacings))
            if mean_spacing <= 1.0:
                continue
            max_dev = float(np.max(np.abs(spacings - mean_spacing)) / mean_spacing)
            if max_dev > max_spacing_dev_ratio:
                continue
            score = self._score_group(group)
            if score < best_score:
                best_score = score
                best_group = group
        return best_group

    def locate(self, frame: np.ndarray) -> LocatorResult:
        try:
            x0, y0, x1, y1 = self._search_bounds(frame.shape)
        except ValueError as exc:
            return LocatorResult(regions=[], status="failed", confidence=0.0, debug_info={"error": str(exc), "locator_type": "slot_based"})

        search = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        dark_threshold = int(self.slot_detection.get("dark_threshold", 90))
        dark_mask = (v < dark_threshold).astype(np.uint8) * 255

        kernel_size = int(self.slot_detection.get("morphology_kernel_size", 3))
        kernel = np.ones((max(1, kernel_size), max(1, kernel_size)), dtype=np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = float(self.slot_detection.get("min_area", 40))
        max_area = float(self.slot_detection.get("max_area", 3000))
        min_height = int(self.slot_detection.get("min_height", 15))
        max_width = int(self.slot_detection.get("max_width", 80))
        min_aspect_ratio = float(self.slot_detection.get("min_aspect_ratio", 1.2))

        candidates: list[tuple[int, int, int, int]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h < min_height or w > max_width:
                continue
            aspect_ratio = h / max(1.0, float(w))
            if aspect_ratio < min_aspect_ratio:
                continue
            candidates.append((x + x0, y + y0, w, h))

        candidates = sorted(candidates, key=lambda b: b[0])
        selected = self._select_plausible_five(candidates)
        debug = {
            "locator_type": "slot_based",
            "search_region": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
            "all_candidates": [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h in candidates],
            "selected_candidates": [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h in selected],
            "candidate_count": len(candidates),
            "selected_count": len(selected),
        }
        if len(selected) != self.expected_led_count:
            return LocatorResult(regions=[], status="failed", confidence=0.0, debug_info=debug)

        roi_w = int(self.roi_cfg.get("width", 40))
        roi_h = int(self.roi_cfg.get("height", 40))
        offset_x = int(self.roi_cfg.get("offset_x", 0))
        offset_y = int(self.roi_cfg.get("offset_y", 0))
        h, w = frame.shape[:2]

        regions: list[LEDRegion] = []
        for i, (sx, sy, sw, sh) in enumerate(selected, start=1):
            cx = sx + sw // 2 + offset_x
            cy = sy + sh // 2 + offset_y
            rx = int(np.clip(cx - roi_w // 2, 0, max(0, w - 1)))
            ry = int(np.clip(cy - roi_h // 2, 0, max(0, h - 1)))
            rw = int(min(roi_w, w - rx))
            rh = int(min(roi_h, h - ry))
            regions.append(LEDRegion(led_id=f"led_{i}", x=rx, y=ry, width=rw, height=rh, confidence=1.0, source="slot_based"))

        return LocatorResult(regions=regions, status="ok", confidence=1.0, debug_info=debug)
