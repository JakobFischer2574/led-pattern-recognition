from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def draw_led_debug_overlay(frame: np.ndarray, led_metrics: list[dict[str, Any]], rois: dict[str, dict[str, int]]) -> np.ndarray:
    canvas = frame.copy()
    for idx, (name, roi) in enumerate(rois.items()):
        x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
        metrics = led_metrics[idx]
        state = metrics["state"]
        color = (0, 255, 0) if state == 1 else (0, 0, 255)
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
) -> np.ndarray:
    """Erstellt ein Debug-Bild mit Originalframe (links) und Info-Panel (rechts)."""
    base = frame.copy()
    h, w = base.shape[:2]
    panel = np.full((h, info_panel_width, 3), 25, dtype=np.uint8)

    for led in debug_info:
        x = int(led["x"])
        y = int(led["y"])
        rw = int(led["width"])
        rh = int(led["height"])
        state = int(led["state"])
        color = (0, 200, 0) if state == 1 else (0, 0, 220)
        cv2.rectangle(base, (x, y), (x + rw, y + rh), color, 2)

    canvas = np.hstack([base, panel])
    x0 = w + 12
    line_h = 20
    y0 = 28

    def draw_text(line: str, y: int, color: tuple[int, int, int] = (245, 245, 245), scale: float = 0.5) -> None:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(canvas, (x0 - 4, y - th - 4), (x0 + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(canvas, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    draw_text("Classic-CV Debug", y0, (255, 220, 120), 0.6)
    y0 += line_h
    draw_text(f"Frame: {frame_name}", y0)
    y0 += line_h
    draw_text(f"Layout: {Path(layout_path).name}", y0)
    y0 += line_h
    draw_text(f"Config: {Path(config_path).name}", y0)
    y0 += line_h + 6

    for led in debug_info:
        state = int(led["state"])
        color = (0, 200, 0) if state == 1 else (0, 0, 220)
        draw_text(
            f"{led['led_id']}: {'ON' if state == 1 else 'OFF'} conf={float(led['confidence']):.3f}",
            y0,
            color,
            0.5,
        )
        y0 += line_h
        draw_text(
            f"mean={float(led['mean_brightness']):.1f} max={float(led['max_brightness']):.1f} ratio={float(led['bright_pixel_ratio']):.3f}",
            y0,
            (220, 220, 220),
            0.45,
        )
        y0 += line_h
        draw_text(
            f"g_ratio={float(led['green_pixel_ratio']):.3f} g_max={float(led['max_green_score']):.1f} g_cc={float(led['largest_green_component_area']):.1f}",
            y0,
            (160, 255, 160),
            0.45,
        )
        y0 += line_h
        draw_text(f"roi=({led['x']},{led['y']},{led['width']},{led['height']})", y0, (180, 180, 180), 0.42)
        y0 += line_h + 4
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
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    composed = draw_detection_debug_image(frame, debug_info, frame_name, layout_path, config_path)
    out_path = output_dir / frame_name
    save_debug_image(out_path, composed)

    if save_roi_crops:
        stem = Path(frame_name).stem
        ext = Path(frame_name).suffix or ".png"
        crop_dir = output_dir / f"{stem}_roi_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        for led in debug_info:
            x, y = int(led["x"]), int(led["y"])
            rw, rh = int(led["width"]), int(led["height"])
            crop = frame[max(0, y):max(0, y + rh), max(0, x):max(0, x + rw)]
            if crop.size == 0:
                continue
            enlarged = cv2.resize(crop, (rw * max(1, roi_scale), rh * max(1, roi_scale)), interpolation=cv2.INTER_NEAREST)
            color = (0, 200, 0) if int(led["state"]) == 1 else (0, 0, 220)
            cv2.rectangle(enlarged, (0, 0), (enlarged.shape[1] - 1, enlarged.shape[0] - 1), color, 2)
            crop_name = crop_dir / f"{led['led_id']}_{'on' if int(led['state']) == 1 else 'off'}{ext}"
            save_debug_image(crop_name, enlarged)


    if save_masks:
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(frame_name).stem
        for led in debug_info:
            mask = led.get("green_mask")
            if mask is None:
                continue
            mask_u8 = np.asarray(mask, dtype=np.uint8)
            mask_name = mask_dir / f"{stem}_{led['led_id']}_mask.jpg"
            save_debug_image(mask_name, mask_u8)
    return out_path
