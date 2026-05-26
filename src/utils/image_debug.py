from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _load_thresholds_from_config(config_path: str | Path) -> dict[str, Any] | None:
    try:
        import yaml  # type: ignore
    except Exception:
        return None

    try:
        with Path(config_path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else None
    except Exception:
        return None


def _get_nested_value(data: dict[str, Any] | None, candidate_paths: list[tuple[str, ...]]) -> Any:
    if not isinstance(data, dict):
        return "n/a"

    for path in candidate_paths:
        cur: Any = data
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return "n/a"


def _format_value(v: Any, precision: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def draw_led_debug_overlay(frame: np.ndarray, led_metrics: list[dict[str, Any]], rois: dict[str, dict[str, int]]) -> np.ndarray:
    canvas = frame.copy()
    for idx, (name, roi) in enumerate(rois.items()):
        x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
        metrics = led_metrics[idx] if idx < len(led_metrics) else {"state": -1, "mean": 0.0, "max": 0.0, "ratio": 0.0}
        state = metrics["state"]
        color = (0, 255, 0) if state == 1 else ((0, 0, 255) if state == 0 else (0, 140, 255))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        text = f"{name}:{state} m={metrics['mean']:.1f} mx={metrics['max']:.1f} r={metrics['ratio']:.2f}"
        cv2.putText(canvas, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas


def save_debug_image(path: str | Path, image: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise IOError(f"Debug-Bild konnte nicht gespeichert werden: {output}")


def draw_detection_debug_image(
    frame: np.ndarray,
    debug_info: list[dict[str, Any]],
    frame_name: str,
    layout_path: str | Path,
    config_path: str | Path,
    info_panel_width: int = 520,
    locator_data: dict[str, Any] | None = None,
    locator_type: str = "n/a",
    locator_status: str = "ok",
    locator_confidence: float = 1.0,
) -> np.ndarray:
    base = frame.copy()
    h, w = base.shape[:2]
    panel = np.full((h, info_panel_width, 3), 25, dtype=np.uint8)

    locator_data = locator_data or {}
    search_region = locator_data.get("search_region")
    if search_region:
        x, y = int(search_region["x"]), int(search_region["y"])
        rw, rh = int(search_region["width"]), int(search_region["height"])
        cv2.rectangle(base, (x, y), (x + rw, y + rh), (255, 255, 0), 2)

    for c in locator_data.get("all_candidates", []):
        cv2.rectangle(base, (int(c["x"]), int(c["y"])), (int(c["x"] + c["width"]), int(c["y"] + c["height"])), (0, 255, 255), 1)

    for c in locator_data.get("selected_candidates", []):
        cv2.rectangle(base, (int(c["x"]), int(c["y"])), (int(c["x"] + c["width"]), int(c["y"] + c["height"])), (255, 0, 0), 2)

    for led in debug_info:
        x = int(led["x"])
        y = int(led["y"])
        rw = int(led["width"])
        rh = int(led["height"])
        state = int(led["state"])
        color = (0, 200, 0) if state == 1 else (0, 0, 220)
        cv2.rectangle(base, (x, y), (x + rw, y + rh), color, 2)

    if locator_status == "failed":
        cv2.putText(base, "LOCATOR FAILED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    elif locator_status == "tracked_fallback":
        cv2.putText(base, "TRACKED FALLBACK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3, cv2.LINE_AA)

    canvas = np.hstack([base, panel])
    x0 = w + 12
    line_h = 18
    y0 = 28
    thresholds_cfg = _load_thresholds_from_config(config_path)

    threshold_specs: list[tuple[str, list[tuple[str, ...]]]] = [
        ("locator.roi.width", [("locator", "roi", "width")]),
        ("locator.roi.height", [("locator", "roi", "height")]),
        ("locator.roi.offset_x", [("locator", "roi", "offset_x")]),
        ("locator.roi.offset_y", [("locator", "roi", "offset_y")]),
    ]

    def draw_text(line: str, y: int, color: tuple[int, int, int] = (245, 245, 245), scale: float = 0.5) -> None:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(canvas, (x0 - 4, y - th - 4), (x0 + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(canvas, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    draw_text("Classic-CV Debug", y0, (255, 220, 120), 0.6); y0 += line_h
    draw_text(f"Frame: {frame_name}", y0); y0 += line_h
    draw_text(f"Layout: {Path(layout_path).name}", y0); y0 += line_h
    draw_text(f"Config: {Path(config_path).name}", y0); y0 += line_h
    draw_text(f"locator_type: {locator_type}", y0); y0 += line_h
    draw_text(f"locator_status: {locator_status}", y0); y0 += line_h
    draw_text(f"locator_conf: {locator_confidence:.3f}", y0); y0 += line_h
    draw_text(f"candidates: {locator_data.get('candidate_count', 0)}", y0); y0 += line_h
    draw_text(f"selected: {locator_data.get('selected_count', 0)}", y0); y0 += line_h
    if "fallback_counter" in locator_data:
        draw_text(f"fallback_counter: {locator_data['fallback_counter']}", y0); y0 += line_h
    draw_text("thresholds:", y0, (255, 220, 120), 0.5); y0 += line_h
    if thresholds_cfg is None:
        draw_text("thresholds=n/a", y0, (180, 180, 180), 0.42); y0 += line_h
    else:
        for label, candidates in threshold_specs:
            draw_text(f"{label}={_format_value(_get_nested_value(thresholds_cfg, candidates))}", y0, (180, 180, 180), 0.40)
            y0 += line_h
    y0 += 6

    for led in debug_info:
        state_raw = led.get("state", -1)
        state = int(state_raw) if isinstance(state_raw, (int, np.integer, float)) else -1
        state_str = "ON" if state == 1 else ("OFF" if state == 0 else "UNKNOWN")
        color = (0, 200, 0) if state == 1 else (0, 0, 220)
        draw_text(f"{_format_value(led.get('led_id'))}: {state_str} conf={_format_value(led.get('confidence'))}", y0, color, 0.45); y0 += line_h
        draw_text(
            f"roi=({_format_value(led.get('x'))},{_format_value(led.get('y'))},{_format_value(led.get('width'))},{_format_value(led.get('height'))})",
            y0,
            (180, 180, 180),
            0.40,
        ); y0 += line_h
        draw_text(
            f"bright mean={_format_value(led.get('mean_brightness'))} max={_format_value(led.get('max_brightness'))} ratio={_format_value(led.get('bright_pixel_ratio'))}",
            y0,
            (170, 210, 255),
            0.38,
        ); y0 += line_h
        draw_text(
            f"green ratio={_format_value(led.get('green_pixel_ratio'))} max_score={_format_value(led.get('max_green_score'))} largest={_format_value(led.get('largest_green_component_area'))}",
            y0,
            (140, 255, 140),
            0.38,
        ); y0 += line_h
        draw_text(
            f"areas g={_format_value(led.get('green_area'))} w={_format_value(led.get('white_core_area'))} valid_w={_format_value(led.get('valid_white_core_area'))}",
            y0,
            (255, 220, 170),
            0.38,
        ); y0 += line_h
        draw_text(
            f"combined area={_format_value(led.get('combined_led_area'))} largest={_format_value(led.get('combined_largest_component_area'))}",
            y0,
            (255, 220, 170),
            0.38,
        ); y0 += line_h + 3
        if y0 > h - 20:
            break

    return canvas


def save_detection_debug_artifacts(
    output_dir: str | Path,
    frame: np.ndarray,
    frame_name: str,
    layout_path: str | Path,
    config_path: str | Path,
    debug_info: list[dict[str, Any]],
    save_roi_crops: bool = False,
    roi_scale: int = 4,
    save_masks: bool = False,
    locator_data: dict[str, Any] | None = None,
    locator_type: str = "n/a",
    locator_status: str = "ok",
    locator_confidence: float = 1.0,
    info_panel_width: int = 720,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    composed = draw_detection_debug_image(
        frame, debug_info, frame_name, layout_path, config_path,
        info_panel_width=info_panel_width,
        locator_data=locator_data, locator_type=locator_type,
        locator_status=locator_status, locator_confidence=locator_confidence,
    )
    out_path = output_dir / frame_name
    save_debug_image(out_path, composed)
    return out_path
