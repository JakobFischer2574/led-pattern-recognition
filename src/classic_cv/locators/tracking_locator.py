from __future__ import annotations

from src.classic_cv.locators.base_led_locator import BaseLEDLocator, LEDRegion, LocatorResult


class TrackingLEDLocator(BaseLEDLocator):
    """Wrapper für auto-Locator mit kurzem Tracking-Fallback auf letzte valide auto-Regionen."""

    def __init__(
        self,
        base_locator: BaseLEDLocator,
        enabled: bool = True,
        max_tracking_fallback_frames: int = 5,
        fallback_confidence: float = 0.5,
    ) -> None:
        self.base_locator = base_locator
        self.enabled = bool(enabled)
        self.max_tracking_fallback_frames = int(max_tracking_fallback_frames)
        self.fallback_confidence = float(fallback_confidence)
        self.last_valid_regions: list[LEDRegion] = []
        self.fallback_counter = 0

    def locate(self, frame):
        result = self.base_locator.locate(frame)
        if result.status == "ok" and result.regions:
            self.last_valid_regions = [LEDRegion(**vars(r)) for r in result.regions]
            self.fallback_counter = 0
            result.debug_info["fallback_counter"] = self.fallback_counter
            return result

        if (
            self.enabled
            and self.last_valid_regions
            and self.fallback_counter < self.max_tracking_fallback_frames
        ):
            self.fallback_counter += 1
            return LocatorResult(
                regions=[LEDRegion(**vars(r)) for r in self.last_valid_regions],
                status="tracked_fallback",
                confidence=self.fallback_confidence,
                debug_info={
                    **result.debug_info,
                    "fallback_counter": self.fallback_counter,
                    "fallback_reason": "slot_locator_failed",
                },
            )

        return LocatorResult(
            regions=[],
            status="failed",
            confidence=0.0,
            debug_info={**result.debug_info, "fallback_counter": self.fallback_counter},
        )
