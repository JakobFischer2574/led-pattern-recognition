def classify_led_state(features: dict[str, float], thresholds: dict[str, float]) -> tuple[int, float]:
    largest_component_area = float(
        features.get(
            "combined_largest_component_area",
            features.get("largest_green_component_area", 0.0)
        )
    )

    max_brightness = float(
        features.get(
            "max_brightness",
            features.get("bright_max", 0.0)
        )
    )

    valid_white_core_area = float(
        features.get("valid_white_core_area", 0.0)
    )

    min_green_pixel_ratio = float(thresholds.get("min_green_pixel_ratio", 0.08))
    min_max_green_score = float(thresholds.get("min_max_green_score", 180.0))
    min_largest_green_component_area = float(thresholds.get("min_largest_green_component_area", 100.0))

    min_valid_white_core_area = float(thresholds.get("min_valid_white_core_area", 8.0))
    min_bright_max = float(
        thresholds.get(
            "min_bright_max",
            thresholds.get("min_max_brightness", 230.0)
        )
    )
    min_combined_component_area = float(thresholds.get("min_combined_component_area", 80.0))

    green_checks = [
        float(features.get("green_pixel_ratio", 0.0)) >= min_green_pixel_ratio,
        float(features.get("max_green_score", 0.0)) >= min_max_green_score,
        float(features.get("largest_green_component_area", 0.0)) >= min_largest_green_component_area,
    ]

    white_core_checks = [
        valid_white_core_area >= min_valid_white_core_area,
        max_brightness >= min_bright_max,
        largest_component_area >= min_combined_component_area,
    ]

    green_confidence = sum(float(v) for v in green_checks) / len(green_checks)
    white_confidence = sum(float(v) for v in white_core_checks) / len(white_core_checks)

    green_on = all(green_checks)
    white_core_on = all(white_core_checks)

    if green_on and white_core_on:
        confidence = green_confidence + white_confidence  # max. 2.0
    elif green_on or white_core_on:
        confidence = max(green_confidence, white_confidence)  # max. 1.0
    else:
        confidence = max(green_confidence, white_confidence)  # unter 1.0

    return (1 if green_on or white_core_on else 0), confidence