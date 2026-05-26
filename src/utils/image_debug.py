from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


Color = tuple[int, int, int]

COLOR_ON: Color = (0, 200, 0)
COLOR_OFF: Color = (0, 0, 220)
COLOR_UNKNOWN: Color = (0, 180, 220)
COLOR_SEARCH_REGION: Color = (0, 220, 220)
COLOR_CANDIDATE: Color = (0, 210, 255)
COLOR_SELECTED_SLOT: Color = (255, 170, 0)
COLOR_TEXT: Color = (245, 245, 245)
COLOR_MUTED: Color = (180, 180, 180)
COLOR_PANEL_BG: Color = (25, 25, 25)


def _as_dict(value: Any) -> dict[str, Any]:
    """Return a dictionary for dicts/dataclasses/objects without mutating the input."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _state_color(state: Any) -> Color:
    state_int = _to_int(state, -1)
    if state_int == 1:
        return COLOR_ON
    if state_int == 0:
        return COLOR_OFF
    return COLOR_UNKNOWN


def _state_label(state: Any) -> str:
    state_int = _to_int(state, -1)
    if state_int == 1:
        return "ON"
    if state_int == 0:
        return "OFF"
    return "UNKNOWN"


def _clip_rect(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(image_width - 1, x))
    y1 = max(0, min(image_height - 1, y))
    x2 = max(0, min(image_width - 1, x + width))
    y2 = max(0, min(image_height - 1, y + height))
    return x1, y1, x2, y2


def _extract_rect(obj: Any, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    """Extract a rectangle from common bbox/ROI representations.

    Supported formats:
    - {"x": ..., "y": ..., "width": ..., "height": ...}
    - {"x": ..., "y": ..., "w": ..., "h": ...}
    - {"bbox": [x, y, w, h]}
    - {"rect": [x, y, w, h]}
    - dataclasses/objects with the same attributes
    """
    data = _as_dict(obj)
    if not data:
        return None

    bbox = data.get("bbox") or data.get("rect")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x, y, width, height = (_to_int(bbox[0]), _to_int(bbox[1]), _to_int(bbox[2]), _to_int(bbox[3]))
        return _clip_rect(x, y, width, height, image_width, image_height)

    x = data.get("x")
    y = data.get("y")
    width = data.get("width", data.get("w"))
    height = data.get("height", data.get("h"))
    if x is None or y is None or width is None or height is None:
        return None
    return _clip_rect(_to_int(x), _to_int(y), _to_int(width), _to_int(height), image_width, image_height)


def _extract_search_region(search_region: Any, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    data = _as_dict(search_region)
    if not data:
        return None

    rect = _extract_rect(data, image_width, image_height)
    if rect is not None:
        return rect

    if all(key in data for key in ("x_min_ratio", "x_max_ratio", "y_min_ratio", "y_max_ratio")):
        x = int(float(data["x_min_ratio"]) * image_width)
        y = int(float(data["y_min_ratio"]) * image_height)
        width = int((float(data["x_max_ratio"]) - float(data["x_min_ratio"])) * image_width)
        height = int((float(data["y_max_ratio"]) - float(data["y_min_ratio"])) * image_height)
        return _clip_rect(x, y, width, height, image_width, image_height)

    return None


def _draw_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Color = COLOR_TEXT,
    scale: float = 0.5,
    thickness: int = 1,
    with_background: bool = True,
) -> None:
    if with_background:
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(
            canvas,
            (x - 4, y - text_height - baseline - 4),
            (x + text_width + 4, y + baseline + 4),
            (0, 0, 0),
            -1,
        )
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_rect_with_label(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    color: Color,
    thickness: int = 2,
    scale: float = 0.45,
) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    if label:
        _draw_text(canvas, label, x1, max(18, y1 - 6), color, scale=scale, thickness=1)


def _normalize_led_debug_info(debug_info: Any) -> list[dict[str, Any]]:
    if debug_info is None:
        return []
    if isinstance(debug_info, list):
        return [_as_dict(item) for item in debug_info]
    if isinstance(debug_info, dict):
        for key in ("leds", "led_debug_info", "led_metrics", "classification_debug"):
            value = debug_info.get(key)
            if isinstance(value, list):
                return [_as_dict(item) for item in value]
    return []


def _normalize_locator_debug_info(debug_info: Any, locator_debug_info: Any | None = None) -> dict[str, Any]:
    if locator_debug_info is not None:
        return _as_dict(locator_debug_info)
    if isinstance(debug_info, dict):
        for key in ("locator", "locator_debug", "locator_debug_info"):
            value = debug_info.get(key)
            if value is not None:
                return _as_dict(value)
    return {}


def _normalize_locator_result(locator_result: Any | None) -> dict[str, Any]:
    result = _as_dict(locator_result)
    if not result and locator_result is not None:
        result = {
            "regions": _get(locator_result, "regions", []),
            "status": _get(locator_result, "status", "unknown"),
            "confidence": _get(locator_result, "confidence", 0.0),
            "debug_info": _get(locator_result, "debug_info", {}),
        }
    return result


def draw_led_debug_overlay(frame: np.ndarray, led_metrics: list[dict[str, Any]], rois: dict[str, dict[str, int]]) -> np.ndarray:
    """Legacy helper: draw simple ROI overlays directly on the frame."""
    canvas = frame.copy()
    for idx, (name, roi) in enumerate(rois.items()):
        if idx >= len(led_metrics):
            continue
        x, y, width, height = roi["x"], roi["y"], roi["width"], roi["height"]
        metrics = led_metrics[idx]
        state = metrics.get("state", -1)
        color = _state_color(state)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 2)
        text = (
            f"{name}:{state} "
            f"m={_to_float(metrics.get('mean', metrics.get('mean_brightness'))):.1f} "
            f"mx={_to_float(metrics.get('max', metrics.get('max_brightness'))):.1f} "
            f"r={_to_float(metrics.get('ratio', metrics.get('bright_pixel_ratio'))):.2f}"
        )
        _draw_text(canvas, text, x, max(20, y - 8), color, scale=0.4)
    return canvas


def save_debug_image(path: str | Path, image: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise IOError(f"Debug-Bild konnte nicht gespeichert werden: {output}")


def _draw_locator_overlay(base: np.ndarray, locator_debug_info: dict[str, Any], locator_result: Any | None = None) -> None:
    """Draw search region, all candidates, selected candidates and final regions."""
    image_height, image_width = base.shape[:2]
    result = _normalize_locator_result(locator_result)
    result_debug = _as_dict(result.get("debug_info"))
    combined_debug = {**locator_debug_info, **result_debug}

    search_region = (
        combined_debug.get("search_region")
        or combined_debug.get("search_region_px")
        or combined_debug.get("search_region_rect")
    )
    search_rect = _extract_search_region(search_region, image_width, image_height)
    if search_rect is not None:
        _draw_rect_with_label(base, search_rect, "search", COLOR_SEARCH_REGION, thickness=2, scale=0.45)

    candidates = (
        combined_debug.get("candidates")
        or combined_debug.get("all_candidates")
        or combined_debug.get("slot_candidates")
        or []
    )
    for idx, candidate in enumerate(candidates):
        rect = _extract_rect(candidate, image_width, image_height)
        if rect is not None:
            _draw_rect_with_label(base, rect, f"cand {idx + 1}", COLOR_CANDIDATE, thickness=1, scale=0.38)

    selected = (
        combined_debug.get("selected_candidates")
        or combined_debug.get("selected_slots")
        or combined_debug.get("selected_slot_candidates")
        or []
    )
    for idx, slot in enumerate(selected):
        rect = _extract_rect(slot, image_width, image_height)
        if rect is not None:
            _draw_rect_with_label(base, rect, f"slot {idx + 1}", COLOR_SELECTED_SLOT, thickness=2, scale=0.42)

    regions = result.get("regions") or combined_debug.get("regions") or combined_debug.get("led_regions") or []
    for idx, region in enumerate(regions):
        region_data = _as_dict(region)
        rect = _extract_rect(region_data, image_width, image_height)
        if rect is None:
            continue
        led_id = str(region_data.get("led_id", f"led_{idx + 1}"))
        _draw_rect_with_label(base, rect, led_id, (255, 220, 120), thickness=2, scale=0.45)


def draw_detection_debug_image(
    frame: np.ndarray,
    debug_info: list[dict[str, Any]] | dict[str, Any],
    frame_name: str,
    layout_path: str | Path,
    config_path: str | Path,
    info_panel_width: int = 560,
    locator_debug_info: dict[str, Any] | None = None,
    locator_status: str | None = None,
    locator_confidence: float | None = None,
    locator_type: str | None = None,
    locator_result: Any | None = None,
) -> np.ndarray:
    """Create a combined debug image for LED localization and state classification.

    The function is backwards-compatible with the old call signature and accepts optional
    locator information for the SlotBasedLEDLocator.
    """
    base = frame.copy()
    image_height, image_width = base.shape[:2]
    panel = np.full((image_height, info_panel_width, 3), COLOR_PANEL_BG, dtype=np.uint8)

    leds = _normalize_led_debug_info(debug_info)
    locator_debug = _normalize_locator_debug_info(debug_info, locator_debug_info)
    result = _normalize_locator_result(locator_result)

    if result:
        locator_status = locator_status or str(result.get("status", "unknown"))
        locator_confidence = locator_confidence if locator_confidence is not None else _to_float(result.get("confidence"), 0.0)
        locator_debug = {**locator_debug, **_as_dict(result.get("debug_info"))}

    _draw_locator_overlay(base, locator_debug, locator_result=result if result else None)

    for led in leds:
        rect = _extract_rect(led, image_width, image_height)
        if rect is None:
            continue
        state = led.get("state", -1)
        color = _state_color(state)
        led_id = str(led.get("led_id", "led"))
        _draw_rect_with_label(base, rect, f"{led_id}:{_state_label(state)}", color, thickness=2, scale=0.45)

    if locator_status == "failed":
        _draw_text(base, "LOCATOR FAILED", 24, 42, COLOR_UNKNOWN, scale=0.9, thickness=2)
    elif locator_status == "tracked_fallback":
        _draw_text(base, "TRACKED FALLBACK", 24, 42, COLOR_SEARCH_REGION, scale=0.8, thickness=2)

    canvas = np.hstack([base, panel])
    x0 = image_width + 12
    y = 28
    line_h = 20

    def panel_text(line: str, color: Color = COLOR_TEXT, scale: float = 0.5, step: int | None = None) -> None:
        nonlocal y
        _draw_text(canvas, line, x0, y, color, scale=scale, thickness=1)
        y += step if step is not None else line_h

    panel_text("Classic-CV Debug", (255, 220, 120), 0.6)
    panel_text(f"Frame: {frame_name}")
    panel_text(f"Image: {image_width}x{image_height}")
    panel_text(f"Layout: {Path(layout_path).name}")
    panel_text(f"Config: {Path(config_path).name}")

    if locator_type or locator_status or locator_debug:
        y += 6
        panel_text("Locator", (255, 220, 120), 0.55)
        panel_text(f"type={locator_type or locator_debug.get('locator_type', 'unknown')}", COLOR_MUTED, 0.45)
        panel_text(f"status={locator_status or locator_debug.get('status', 'unknown')}", COLOR_MUTED, 0.45)
        confidence = locator_confidence if locator_confidence is not None else _to_float(locator_debug.get("confidence"), 0.0)
        panel_text(f"confidence={confidence:.3f}", COLOR_MUTED, 0.45)
        candidate_count = locator_debug.get("candidate_count")
        if candidate_count is None:
            candidate_count = len(locator_debug.get("candidates", locator_debug.get("all_candidates", [])) or [])
        selected_count = locator_debug.get("selected_count")
        if selected_count is None:
            selected_count = len(locator_debug.get("selected_candidates", locator_debug.get("selected_slots", [])) or [])
        panel_text(f"candidates={candidate_count} selected={selected_count}", COLOR_MUTED, 0.45)
        if "fallback_counter" in locator_debug:
            panel_text(f"fallback_counter={locator_debug.get('fallback_counter')}", COLOR_MUTED, 0.45)

    y += 6
    panel_text("LED classification", (255, 220, 120), 0.55)

    if not leds:
        panel_text("No LED debug info available", COLOR_UNKNOWN, 0.45)
        return canvas

    for led in leds:
        if y > image_height - 80:
            panel_text("...", COLOR_MUTED, 0.45)
            break

        state = led.get("state", -1)
        color = _state_color(state)
        led_id = str(led.get("led_id", "led"))
        confidence = _to_float(led.get("confidence"), 0.0)
        panel_text(f"{led_id}: {_state_label(state)} conf={confidence:.3f}", color, 0.5)

        panel_text(
            f"mean={_to_float(led.get('mean_brightness', led.get('mean'))):.1f} "
            f"max={_to_float(led.get('max_brightness', led.get('max'))):.1f} "
            f"ratio={_to_float(led.get('bright_pixel_ratio', led.get('ratio'))):.3f}",
            (220, 220, 220),
            0.43,
        )
        panel_text(
            f"g_ratio={_to_float(led.get('green_pixel_ratio')):.3f} "
            f"g_max={_to_float(led.get('max_green_score')):.1f} "
            f"g_cc={_to_float(led.get('largest_green_component_area')):.1f}",
            (160, 255, 160),
            0.43,
        )

        white_core_area = led.get("white_core_area")
        valid_white_core_area = led.get("valid_white_core_area")
        combined_led_area = led.get("combined_led_area")
        combined_largest_component_area = led.get("combined_largest_component_area")
        if any(value is not None for value in (white_core_area, valid_white_core_area, combined_led_area, combined_largest_component_area)):
            panel_text(
                f"wc={_to_float(white_core_area):.0f} "
                f"vwc={_to_float(valid_white_core_area):.0f} "
                f"comb={_to_float(combined_led_area):.0f} "
                f"cc={_to_float(combined_largest_component_area):.0f}",
                (180, 220, 255),
                0.43,
            )

        panel_text(
            f"roi=({_to_int(led.get('x'))},{_to_int(led.get('y'))},"
            f"{_to_int(led.get('width', led.get('w')))},"
            f"{_to_int(led.get('height', led.get('h')))})",
            COLOR_MUTED,
            0.42,
            step=line_h + 4,
        )

    return canvas


def draw_slot_locator_debug_image(
    frame: np.ndarray,
    locator_result: Any | None = None,
    frame_name: str = "",
    config_path: str | Path = "",
    debug_info: dict[str, Any] | None = None,
    info_panel_width: int = 560,
) -> np.ndarray:
    """Create a debug image focused on SlotBasedLEDLocator output."""
    base = frame.copy()
    image_height, image_width = base.shape[:2]
    panel = np.full((image_height, info_panel_width, 3), COLOR_PANEL_BG, dtype=np.uint8)
    result = _normalize_locator_result(locator_result)
    locator_debug = _as_dict(debug_info)
    if result:
        locator_debug = {**locator_debug, **_as_dict(result.get("debug_info"))}

    _draw_locator_overlay(base, locator_debug, locator_result=result if result else None)

    status = str(result.get("status", locator_debug.get("status", "unknown"))) if result or locator_debug else "unknown"
    confidence = _to_float(result.get("confidence", locator_debug.get("confidence", 0.0))) if result or locator_debug else 0.0
    if status == "failed":
        _draw_text(base, "LOCATOR FAILED", 24, 42, COLOR_UNKNOWN, scale=0.9, thickness=2)
    elif status == "tracked_fallback":
        _draw_text(base, "TRACKED FALLBACK", 24, 42, COLOR_SEARCH_REGION, scale=0.8, thickness=2)

    canvas = np.hstack([base, panel])
    x0 = image_width + 12
    y = 28
    line_h = 20

    def panel_text(line: str, color: Color = COLOR_TEXT, scale: float = 0.5, step: int | None = None) -> None:
        nonlocal y
        _draw_text(canvas, line, x0, y, color, scale=scale, thickness=1)
        y += step if step is not None else line_h

    panel_text("Slot Locator Debug", (255, 220, 120), 0.6)
    panel_text(f"Frame: {frame_name or '-'}")
    panel_text(f"Image: {image_width}x{image_height}")
    if config_path:
        panel_text(f"Config: {Path(config_path).name}")
    y += 6
    panel_text(f"status={status}", COLOR_MUTED, 0.5)
    panel_text(f"confidence={confidence:.3f}", COLOR_MUTED, 0.5)

    candidates = locator_debug.get("candidates", locator_debug.get("all_candidates", [])) or []
    selected = locator_debug.get("selected_candidates", locator_debug.get("selected_slots", [])) or []
    regions = result.get("regions") or locator_debug.get("regions") or locator_debug.get("led_regions") or []
    panel_text(f"candidates={len(candidates)}", COLOR_CANDIDATE, 0.5)
    panel_text(f"selected={len(selected)}", COLOR_SELECTED_SLOT, 0.5)
    panel_text(f"regions={len(regions)}", (255, 220, 120), 0.5)
    if "fallback_counter" in locator_debug:
        panel_text(f"fallback_counter={locator_debug.get('fallback_counter')}", COLOR_MUTED, 0.45)

    y += 6
    for idx, region in enumerate(regions):
        if y > image_height - 40:
            panel_text("...", COLOR_MUTED, 0.45)
            break
        data = _as_dict(region)
        led_id = str(data.get("led_id", f"led_{idx + 1}"))
        panel_text(
            f"{led_id}: x={_to_int(data.get('x'))} y={_to_int(data.get('y'))} "
            f"w={_to_int(data.get('width', data.get('w')))} h={_to_int(data.get('height', data.get('h')))} "
            f"conf={_to_float(data.get('confidence')):.2f}",
            COLOR_MUTED,
            0.42,
        )

    return canvas


def save_detection_debug_artifacts(
    output_dir: str | Path,
    frame: np.ndarray,
    frame_name: str,
    layout_path: str | Path,
    config_path: str | Path,
    debug_info: list[dict[str, Any]] | dict[str, Any],
    save_roi_crops: bool = False,
    roi_scale: int = 4,
    save_masks: bool = False,
    locator_debug_info: dict[str, Any] | None = None,
    locator_status: str | None = None,
    locator_confidence: float | None = None,
    locator_type: str | None = None,
    locator_result: Any | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    leds = _normalize_led_debug_info(debug_info)

    composed = draw_detection_debug_image(
        frame,
        debug_info,
        frame_name,
        layout_path,
        config_path,
        locator_debug_info=locator_debug_info,
        locator_status=locator_status,
        locator_confidence=locator_confidence,
        locator_type=locator_type,
        locator_result=locator_result,
    )
    out_path = output_dir / frame_name
    save_debug_image(out_path, composed)

    if save_roi_crops:
        stem = Path(frame_name).stem
        ext = Path(frame_name).suffix or ".png"
        crop_dir = output_dir / f"{stem}_roi_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        image_height, image_width = frame.shape[:2]
        for led in leds:
            rect = _extract_rect(led, image_width, image_height)
            if rect is None:
                continue
            x1, y1, x2, y2 = rect
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_height, crop_width = crop.shape[:2]
            enlarged = cv2.resize(
                crop,
                (crop_width * max(1, roi_scale), crop_height * max(1, roi_scale)),
                interpolation=cv2.INTER_NEAREST,
            )
            color = _state_color(led.get("state", -1))
            cv2.rectangle(enlarged, (0, 0), (enlarged.shape[1] - 1, enlarged.shape[0] - 1), color, 2)
            led_id = str(led.get("led_id", "led"))
            crop_name = crop_dir / f"{led_id}_{_state_label(led.get('state', -1)).lower()}{ext}"
            save_debug_image(crop_name, enlarged)

    if save_masks:
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(frame_name).stem
        for led in leds:
            led_id = str(led.get("led_id", "led"))
            for mask_key, suffix in (
                ("green_mask", "green"),
                ("white_core_mask", "white_core"),
                ("valid_white_core_mask", "valid_white_core"),
                ("combined_led_mask", "combined_led"),
            ):
                mask = led.get(mask_key)
                if mask is None:
                    continue
                mask_u8 = np.asarray(mask, dtype=np.uint8)
                mask_name = mask_dir / f"{stem}_{led_id}_{suffix}.jpg"
                save_debug_image(mask_name, mask_u8)

    return out_path


def save_slot_locator_debug_artifacts(
    output_dir: str | Path,
    frame: np.ndarray,
    frame_name: str,
    locator_result: Any | None = None,
    config_path: str | Path = "",
    debug_info: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    composed = draw_slot_locator_debug_image(
        frame=frame,
        locator_result=locator_result,
        frame_name=frame_name,
        config_path=config_path,
        debug_info=debug_info,
    )
    out_path = output_dir / frame_name
    save_debug_image(out_path, composed)
    return out_path

