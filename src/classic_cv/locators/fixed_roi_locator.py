from __future__ import annotations

from typing import Any

import numpy as np

from src.classic_cv.locators.base_led_locator import BaseLEDLocator, LEDRegion, LocatorResult


class FixedROILocator(BaseLEDLocator):
    def __init__(self, led_layout: dict[str, dict[str, int]]) -> None:
        self.led_layout = led_layout

    def locate(self, frame: np.ndarray) -> LocatorResult:
        h, w = frame.shape[:2]
        regions: list[LEDRegion] = []
        for led_name in sorted(self.led_layout.keys()):
            roi = self.led_layout[led_name]
            x = int(max(0, min(w - 1, roi["x"])))
            y = int(max(0, min(h - 1, roi["y"])))
            width = int(max(1, min(w - x, roi["width"])))
            height = int(max(1, min(h - y, roi["height"])))
            regions.append(LEDRegion(led_id=led_name, x=x, y=y, width=width, height=height, confidence=1.0, source="fixed_roi"))

        return LocatorResult(
            regions=regions,
            status="ok",
            confidence=1.0,
            debug_info={"locator_type": "fixed_roi", "selected_candidates": []},
        )
