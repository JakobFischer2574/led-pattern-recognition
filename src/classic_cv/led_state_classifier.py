# from __future__ import annotations
#
#
# def classify_led_state(features: dict[str, float], thresholds: dict[str, float]) -> tuple[int, float]:
#     green_checks = [
#         features["green_pixel_ratio"] >= float(thresholds.get("min_green_pixel_ratio", 0.015)),
#         features["max_green_score"] >= float(thresholds.get("min_max_green_score", 40.0)),
#         features["largest_green_component_area"] >= float(thresholds.get("min_largest_green_component_area", 8.0)),
#     ]
#     confidence = sum(float(v) for v in green_checks) / len(green_checks)
#     return (1 if all(green_checks) else 0), confidence



def classify_led_state(features: dict[str, float], thresholds: dict[str, float]) -> tuple[int, float]:
    largest_component_area = float(
        features.get("combined_largest_component_area", features.get("largest_green_component_area", 0.0))
    )

    green_checks = [
        features["green_pixel_ratio"] >= float(thresholds.get("min_green_pixel_ratio", 0.015)),
        features["max_green_score"] >= float(thresholds.get("min_max_green_score", 40.0)),
        features["largest_green_component_area"] >= float(thresholds.get("min_largest_green_component_area", 8.0)),
    ]
    confidence = sum(float(v) for v in green_checks) / len(green_checks)
    return (1 if all(green_checks) else 0), confidence