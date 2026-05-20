from __future__ import annotations


def classify_led_state(features: dict[str, float], thresholds: dict[str, float]) -> tuple[int, float]:
    checks = [
        features["mean_brightness"] >= thresholds["min_mean_brightness"],
        features["max_brightness"] >= thresholds["min_max_brightness"],
        features["bright_pixel_ratio"] >= thresholds["min_bright_pixel_ratio"],
    ]
    confidence = sum(float(v) for v in checks) / len(checks)
    return (1 if all(checks) else 0), confidence
