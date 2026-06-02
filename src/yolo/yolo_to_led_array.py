from __future__ import annotations

from typing import Any

LED_CLASS_TO_STATE = {
    "led_on": 1,
    "led_off": 0,
}


def _state_from_detection(detection: dict[str, Any]) -> int | None:
    class_name = str(detection.get("class_name", "")).strip()
    if class_name in LED_CLASS_TO_STATE:
        return LED_CLASS_TO_STATE[class_name]
    return None


def _confidence_from_detection(detection: dict[str, Any]) -> float:
    try:
        return float(detection.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _x_center_from_detection(detection: dict[str, Any]) -> float:
    try:
        return float(detection.get("x_center", 0.0))
    except (TypeError, ValueError):
        return 0.0


def yolo_boxes_to_led_array(
    detections: list[dict[str, Any]],
    expected_led_count: int = 5,
) -> tuple[list[int], list[float], list[dict[str, Any]]]:
    """Convert YOLO LED detections into a fixed-size LED state array.

    Only the known classes ``led_on`` and ``led_off`` are considered. If YOLO
    returns too many detections, the highest-confidence detections are kept and
    then ordered from left to right by their x-center. Missing LEDs are filled
    with ``-1`` and confidence ``0.0``.
    """
    if expected_led_count <= 0:
        raise ValueError("expected_led_count muss größer als 0 sein")

    known_detections: list[dict[str, Any]] = []
    for detection in detections:
        state = _state_from_detection(detection)
        if state is None:
            continue
        normalized = dict(detection)
        normalized["led_state"] = state
        normalized["confidence"] = _confidence_from_detection(detection)
        normalized["x_center"] = _x_center_from_detection(detection)
        known_detections.append(normalized)

    selected_detections = sorted(
        known_detections,
        key=lambda item: (_confidence_from_detection(item), -_x_center_from_detection(item)),
        reverse=True,
    )[:expected_led_count]
    ordered_detections = sorted(selected_detections, key=_x_center_from_detection)

    led_state = [int(detection["led_state"]) for detection in ordered_detections]
    confidences = [_confidence_from_detection(detection) for detection in ordered_detections]

    missing_count = expected_led_count - len(led_state)
    if missing_count > 0:
        led_state.extend([-1] * missing_count)
        confidences.extend([0.0] * missing_count)

    return led_state, confidences, ordered_detections
