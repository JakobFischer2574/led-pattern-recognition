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
        self.slot_band = config.get("slot_band", {})
        self.projection_detection = config.get("projection_detection", {})
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

    def _relative_bounds(self, bounds: tuple[int, int, int, int], ratios: dict[str, Any]) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bounds
        bw = x1 - x0
        bh = y1 - y0
        rx0 = x0 + int(bw * float(ratios.get("x_min_ratio", 0.0)))
        rx1 = x0 + int(bw * float(ratios.get("x_max_ratio", 1.0)))
        ry0 = y0 + int(bh * float(ratios.get("y_min_ratio", 0.0)))
        ry1 = y0 + int(bh * float(ratios.get("y_max_ratio", 1.0)))
        rx0, rx1 = max(x0, rx0), min(x1, rx1)
        ry0, ry1 = max(y0, ry0), min(y1, ry1)
        if rx1 <= rx0 or ry1 <= ry0:
            return x0, y0, x1, y1
        return rx0, ry0, rx1, ry1

    def _find_projection_peaks(self, column_score: np.ndarray, min_distance: int, min_prominence: float) -> list[int]:
        if column_score.size < 3:
            return []
        peaks: list[tuple[int, float]] = []
        for idx in range(1, column_score.size - 1):
            val = float(column_score[idx])
            if val < float(min_prominence):
                continue
            if val < float(column_score[idx - 1]) or val < float(column_score[idx + 1]):
                continue
            local_min = min(float(column_score[idx - 1]), float(column_score[idx + 1]))
            prominence = val - local_min
            if prominence < float(min_prominence):
                continue
            peaks.append((idx, val))

        peaks = sorted(peaks, key=lambda p: p[1], reverse=True)
        selected: list[int] = []
        for pos, _ in peaks:
            if all(abs(pos - sp) >= min_distance for sp in selected):
                selected.append(pos)
        return sorted(selected)

    def _fit_five_slots(self, peak_xs: list[int]) -> tuple[list[int], float | None, int]:
        if not peak_xs:
            return [], None, 0
        expected = self.expected_led_count
        best_seq: list[int] = []
        best_spacing: float | None = None
        best_score = float("inf")
        reconstructed = 0

        def evaluate(start: float, spacing: float) -> tuple[list[int], float, int]:
            targets = [start + i * spacing for i in range(expected)]
            selected = [int(round(t)) for t in targets]
            residual = 0.0
            matched = 0
            for t in targets:
                nearest = min(peak_xs, key=lambda p: abs(p - t))
                dist = abs(nearest - t)
                tol = max(6.0, spacing * 0.35)
                if dist <= tol:
                    matched += 1
                    residual += dist
            recon = expected - matched
            score = residual + (recon * spacing)
            return selected, score, recon

        for i in range(len(peak_xs)):
            for j in range(i + 1, len(peak_xs)):
                spacing = (peak_xs[j] - peak_xs[i]) / max(1, j - i)
                if spacing <= 1.0:
                    continue
                start = peak_xs[i] - i * spacing
                seq, score, recon = evaluate(start, spacing)
                if score < best_score:
                    best_score, best_seq, best_spacing, reconstructed = score, seq, spacing, recon

        if len(peak_xs) >= expected and not best_seq:
            best_seq = sorted(peak_xs)[:expected]
            best_spacing = float(np.mean(np.diff(best_seq))) if len(best_seq) > 1 else None
            reconstructed = 0

        return best_seq, best_spacing, reconstructed

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
        slot_bounds = (x0, y0, x1, y1)
        if bool(self.slot_band.get("enabled", True)):
            slot_bounds = self._relative_bounds((x0, y0, x1, y1), self.slot_band)
        sx0, sy0, sx1, sy1 = slot_bounds
        slot_img = frame[sy0:sy1, sx0:sx1]

        projection_enabled = bool(self.projection_detection.get("enabled", True))
        selected_centers: list[int] = []
        reconstructed_xs: list[int] = []
        estimated_spacing: float | None = None
        projection_peaks: list[int] = []
        projection_peak_count = 0
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

        if projection_enabled and slot_img.size > 0:
            proj_hsv = cv2.cvtColor(slot_img, cv2.COLOR_BGR2HSV)
            proj_v = proj_hsv[:, :, 2]
            proj_dark_threshold = int(self.projection_detection.get("dark_threshold", 90))
            proj_mask = (proj_v < proj_dark_threshold).astype(np.uint8) * 255
            k_w = max(1, int(self.projection_detection.get("vertical_close_kernel_width", 3)))
            k_h = max(1, int(self.projection_detection.get("vertical_close_kernel_height", 11)))
            close_kernel = np.ones((k_h, k_w), dtype=np.uint8)
            proj_mask = cv2.morphologyEx(proj_mask, cv2.MORPH_CLOSE, close_kernel)
            open_ks = max(1, int(self.projection_detection.get("opening_kernel_size", 3)))
            open_kernel = np.ones((open_ks, open_ks), dtype=np.uint8)
            proj_mask = cv2.morphologyEx(proj_mask, cv2.MORPH_OPEN, open_kernel)

            column_score = (proj_mask > 0).sum(axis=0).astype(np.float32)
            smooth_window = max(1, int(self.projection_detection.get("smoothing_window_px", 7)))
            if smooth_window % 2 == 0:
                smooth_window += 1
            if smooth_window > 1:
                kernel = np.ones((smooth_window,), dtype=np.float32) / smooth_window
                smoothed = np.convolve(column_score, kernel, mode="same")
            else:
                smoothed = column_score
            min_distance = max(1, int(self.projection_detection.get("min_peak_distance_px", 25)))
            min_prominence = float(self.projection_detection.get("min_peak_prominence", 5.0))
            projection_peaks_local = self._find_projection_peaks(smoothed, min_distance, min_prominence)
            projection_peaks = [sx0 + int(p) for p in projection_peaks_local]
            projection_peak_count = len(projection_peaks)

            fitted, estimated_spacing, reconstructed_count = self._fit_five_slots(projection_peaks)
            if len(fitted) == self.expected_led_count and (projection_peak_count >= 3):
                selected_centers = fitted
                peak_set = set(projection_peaks)
                reconstructed_xs = [x for x in fitted if x not in peak_set]
            else:
                reconstructed_count = 0
        else:
            reconstructed_count = 0
        debug = {
            "locator_type": "slot_based",
            "search_region": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
            "slot_band": {"x": sx0, "y": sy0, "width": sx1 - sx0, "height": sy1 - sy0},
            "all_candidates": [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h in candidates],
            "selected_candidates": [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h in selected],
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "projection_peak_count": projection_peak_count,
            "projection_peaks": projection_peaks,
            "estimated_spacing": estimated_spacing,
            "reconstructed_count": reconstructed_count,
            "reconstructed_slots": reconstructed_xs,
            "dark_threshold": int(self.projection_detection.get("dark_threshold", dark_threshold)),
            "min_peak_distance_px": int(self.projection_detection.get("min_peak_distance_px", 25)),
            "smoothing_window_px": int(self.projection_detection.get("smoothing_window_px", 7)),
        }
        if selected_centers:
            selected = [(x - 1, sy0, 2, max(1, sy1 - sy0)) for x in selected_centers]
            debug["selected_candidates"] = [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h in selected]
            debug["selected_count"] = len(selected)

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
