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
        self.local_recovery = config.get("local_recovery", {})
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

    def _extract_candidates(
        self,
        frame: np.ndarray,
        bounds: tuple[int, int, int, int],
        dark_threshold: int,
        min_area: float,
        max_area: float,
        min_height: float,
        max_width: int,
        min_aspect_ratio: float,
    ) -> list[tuple[int, int, int, int]]:
        x0, y0, x1, y1 = bounds
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return []
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        dark_mask = (v < dark_threshold).astype(np.uint8) * 255
        kernel_size = int(self.slot_detection.get("morphology_kernel_size", 3))
        kernel = np.ones((max(1, kernel_size), max(1, kernel_size)), dtype=np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        return sorted(candidates, key=lambda b: b[0] + b[2] / 2.0)

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

        dark_threshold = int(self.slot_detection.get("dark_threshold", 90))
        min_area = float(self.slot_detection.get("min_area", 40))
        max_area = float(self.slot_detection.get("max_area", 3000))
        min_height = float(self.slot_detection.get("min_height", 15))
        max_width = int(self.slot_detection.get("max_width", 80))
        min_aspect_ratio = float(self.slot_detection.get("min_aspect_ratio", 1.2))

        original_candidates = self._extract_candidates(
            frame, (x0, y0, x1, y1), dark_threshold, min_area, max_area, min_height, max_width, min_aspect_ratio
        )
        selected = self._select_plausible_five(original_candidates)

        recovered_candidates: list[tuple[int, int, int, int]] = []
        reconstructed_candidates: list[tuple[int, int, int, int]] = []
        estimated_spacing: float | None = None

        if not selected:
            selected = original_candidates.copy()

        selected = sorted(selected, key=lambda c: c[0] + c[2] / 2.0)

        if (
            bool(self.local_recovery.get("enabled", True))
            and len(selected) < self.expected_led_count
            and len(selected) >= int(self.local_recovery.get("min_candidates", 3))
        ):
            centers = np.array([x + w / 2.0 for x, _, w, _ in selected], dtype=np.float32)
            y_centers = np.array([y + h / 2.0 for _, y, _, h in selected], dtype=np.float32)
            heights = np.array([h for _, _, _, h in selected], dtype=np.float32)
            if centers.size >= 2:
                estimated_spacing = float(np.median(np.diff(centers)))
            if estimated_spacing is not None and estimated_spacing > 2.0:
                anchor = centers[0]
                start = anchor
                for k in range(self.expected_led_count):
                    candidate_start = anchor - k * estimated_spacing
                    if candidate_start < x0 - estimated_spacing or candidate_start > x1 + estimated_spacing:
                        continue
                    expected_xs = [candidate_start + i * estimated_spacing for i in range(self.expected_led_count)]
                    residual = sum(min(abs(cx - ex) for cx in centers) for ex in expected_xs)
                    if residual < sum(min(abs(cx - (start + i * estimated_spacing)) for cx in centers) for i in range(self.expected_led_count)):
                        start = candidate_start

                expected_xs = [start + i * estimated_spacing for i in range(self.expected_led_count)]
                median_h = float(np.median(heights)) if heights.size > 0 else 20.0
                mean_y = float(np.median(y_centers)) if y_centers.size > 0 else float((y0 + y1) / 2)
                rx_factor = float(self.local_recovery.get("search_radius_x_factor", 0.45))
                ry_factor = float(self.local_recovery.get("search_radius_y_factor", 1.0))
                max_recovered = int(self.local_recovery.get("max_recovered_markers", 2))

                for ex in expected_xs:
                    if len(recovered_candidates) >= max_recovered or len(selected) >= self.expected_led_count:
                        break
                    if any(abs((sx + sw / 2.0) - ex) <= max(4.0, estimated_spacing * 0.30) for sx, _, sw, _ in selected):
                        continue
                    rx = int(max(x0, ex - estimated_spacing * rx_factor))
                    rw = int(min(x1, ex + estimated_spacing * rx_factor) - rx)
                    ry = int(max(y0, mean_y - median_h * ry_factor))
                    rh = int(min(y1, mean_y + median_h * ry_factor) - ry)
                    if rw <= 2 or rh <= 2:
                        continue
                    local = self._extract_candidates(
                        frame,
                        (rx, ry, rx + rw, ry + rh),
                        dark_threshold + int(self.local_recovery.get("relaxed_dark_threshold_offset", 15)),
                        min_area * float(self.local_recovery.get("relaxed_min_area_factor", 0.5)),
                        max_area,
                        min_height * float(self.local_recovery.get("relaxed_min_height_factor", 0.7)),
                        max_width,
                        min_aspect_ratio,
                    )
                    if not local:
                        continue
                    best = min(local, key=lambda c: abs((c[0] + c[2] / 2.0) - ex))
                    recovered_candidates.append(best)
                    selected.append(best)
                selected = sorted(selected, key=lambda c: c[0] + c[2] / 2.0)

        # geometric reconstruction for remaining missing markers (only positions)
        if len(selected) >= int(self.geometry.get("min_candidates_for_reconstruction", 3)) and len(selected) < self.expected_led_count:
            centers = np.array([x + w / 2.0 for x, _, w, _ in selected], dtype=np.float32)
            y_centers = np.array([y + h / 2.0 for _, y, _, h in selected], dtype=np.float32)
            widths = np.array([w for _, _, w, _ in selected], dtype=np.float32)
            heights = np.array([h for _, _, _, h in selected], dtype=np.float32)
            if centers.size >= 2:
                estimated_spacing = float(np.median(np.diff(centers)))
            if estimated_spacing is not None and estimated_spacing > 2.0:
                start = float(np.min(centers))
                expected_xs = [start + i * estimated_spacing for i in range(self.expected_led_count)]
                for ex in expected_xs:
                    if len(selected) >= self.expected_led_count:
                        break
                    if any(abs((sx + sw / 2.0) - ex) <= max(4.0, estimated_spacing * 0.30) for sx, _, sw, _ in selected):
                        continue
                    rw = int(max(2.0, np.median(widths)))
                    rh = int(max(6.0, np.median(heights)))
                    cy = int(np.median(y_centers))
                    rx = int(np.clip(ex - rw / 2.0, x0, max(x0, x1 - rw)))
                    ry = int(np.clip(cy - rh / 2.0, y0, max(y0, y1 - rh)))
                    rec = (rx, ry, rw, rh)
                    reconstructed_candidates.append(rec)
                    selected.append(rec)
                selected = sorted(selected, key=lambda c: c[0] + c[2] / 2.0)[: self.expected_led_count]

        if estimated_spacing is None and len(selected) >= 2:
            estimated_spacing = float(np.median(np.diff([x + w / 2.0 for x, _, w, _ in selected])))

        original_count = len(original_candidates)
        recovered_count = len(recovered_candidates)
        reconstructed_count = len(reconstructed_candidates)

        # confidence/status
        status = "ok" if len(selected) == self.expected_led_count else "failed"
        confidence = 0.0
        if len(selected) == self.expected_led_count:
            if recovered_count == 0 and reconstructed_count == 0 and len(self._select_plausible_five(original_candidates)) == self.expected_led_count:
                confidence = 1.0
            elif reconstructed_count == 0:
                confidence = 0.9 if recovered_count == 1 else 0.8
            else:
                confidence = 0.75 if reconstructed_count == 1 else 0.65

        measured_markers = selected[:]
        if reconstructed_candidates:
            rec_ids = {(x, y, w, h) for x, y, w, h in reconstructed_candidates}
            measured_markers = [c for c in selected if c not in rec_ids]

        roi_w = int(self.roi_cfg.get("width", 40))
        roi_h = int(self.roi_cfg.get("height", 40))
        offset_x = int(self.roi_cfg.get("offset_x", 0))
        offset_y = int(self.roi_cfg.get("offset_y", 0))
        median_marker_width: float | None = None
        median_marker_height: float | None = None
        dynamic_roi_width = roi_w
        dynamic_roi_height = roi_h
        dynamic_offset_y = offset_y
        if bool(self.roi_cfg.get("dynamic_size_enabled", False)) and measured_markers:
            median_marker_width = float(np.median([m[2] for m in measured_markers]))
            median_marker_height = float(np.median([m[3] for m in measured_markers]))
            dynamic_roi_width = int(np.clip(
                median_marker_width * float(self.roi_cfg.get("width_factor", 5.0)),
                float(self.roi_cfg.get("min_width", 28)),
                float(self.roi_cfg.get("max_width", 80)),
            ))
            dynamic_roi_height = int(np.clip(
                median_marker_height * float(self.roi_cfg.get("height_factor", 1.5)),
                float(self.roi_cfg.get("min_height", 28)),
                float(self.roi_cfg.get("max_height", 80)),
            ))
            dynamic_offset_y = int(median_marker_height * float(self.roi_cfg.get("offset_y_factor", -0.6)))

        h, w = frame.shape[:2]
        regions: list[LEDRegion] = []
        if len(selected) == self.expected_led_count:
            for i, (sx, sy, sw, sh) in enumerate(selected, start=1):
                cx = sx + sw // 2 + offset_x
                cy = sy + sh // 2 + dynamic_offset_y
                rx = int(np.clip(cx - dynamic_roi_width // 2, 0, max(0, w - 1)))
                ry = int(np.clip(cy - dynamic_roi_height // 2, 0, max(0, h - 1)))
                rw = int(min(dynamic_roi_width, w - rx))
                rh = int(min(dynamic_roi_height, h - ry))
                regions.append(LEDRegion(led_id=f"led_{i}", x=rx, y=ry, width=rw, height=rh, confidence=confidence, source="slot_based"))

        debug = {
            "locator_type": "slot_based",
            "search_region": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
            "original_candidates": [{"x": x, "y": y, "width": ww, "height": hh} for x, y, ww, hh in original_candidates],
            "recovered_candidates": [{"x": x, "y": y, "width": ww, "height": hh} for x, y, ww, hh in recovered_candidates],
            "reconstructed_candidates": [{"x": x, "y": y, "width": ww, "height": hh} for x, y, ww, hh in reconstructed_candidates],
            "selected_candidates": [{"x": x, "y": y, "width": ww, "height": hh} for x, y, ww, hh in selected],
            "candidate_count": original_count,
            "selected_count": len(selected),
            "recovered_count": recovered_count,
            "reconstructed_count": reconstructed_count,
            "estimated_spacing": estimated_spacing,
            "median_marker_width": median_marker_width,
            "median_marker_height": median_marker_height,
            "dynamic_roi_width": dynamic_roi_width,
            "dynamic_roi_height": dynamic_roi_height,
            "dynamic_offset_y": dynamic_offset_y,
            "geometry_score": None,
            "locator_status": status,
            "locator_confidence": confidence,
            "dark_threshold": dark_threshold,
        }

        if status != "ok":
            return LocatorResult(regions=[], status="failed", confidence=0.0, debug_info=debug)
        return LocatorResult(regions=regions, status=status, confidence=confidence, debug_info=debug)
