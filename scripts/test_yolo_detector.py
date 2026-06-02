from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detectors.base_detector import DetectionResult
from src.detectors.yolo_detector import YOLODetector
from src.utils.config_loader import load_yaml_config


_BOX_COLORS = {
    "led_on": (0, 255, 0),
    "led_off": (0, 0, 255),
    "unknown": (0, 255, 255),
}

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def iter_frames(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input-Pfad nicht gefunden: {path}")
    if path.is_file():
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"Input-Datei ist keine unterstützte Bilddatei: {path}")
        return [path]
    frames = sorted([p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES])
    if not frames:
        raise ValueError(f"Keine Bilddateien in Ordner gefunden: {path}")
    return frames


def draw_debug_image(frame: Any, result: DetectionResult) -> Any:
    debug_image = frame.copy()
    for index, detection in enumerate(result.debug_info.get("ordered_detections", []), start=1):
        x1 = int(round(float(detection["x1"])))
        y1 = int(round(float(detection["y1"])))
        x2 = int(round(float(detection["x2"])))
        y2 = int(round(float(detection["y2"])))
        class_name = str(detection.get("class_name", "unknown"))
        confidence = float(detection.get("confidence", 0.0))
        color = _BOX_COLORS.get(class_name, _BOX_COLORS["unknown"])
        label = f"LED {index}: {class_name} {confidence:.2f}"

        cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            debug_image,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return debug_image


def save_debug_image(frame: Any, result: DetectionResult, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    debug_image = draw_debug_image(frame, result)
    if not cv2.imwrite(str(output), debug_image):
        raise OSError(f"Debug-Bild konnte nicht geschrieben werden: {output}")


def build_result_row(frame_path: Path, result: DetectionResult) -> dict[str, Any]:
    detections = result.debug_info.get("ordered_detections", [])
    row: dict[str, Any] = {
        "frame_name": frame_path.name,
        "mean_latency_ms": round(result.processing_time_ms, 3),
        "detections": len(detections),
    }

    for i, state in enumerate(result.led_state, start=1):
        row[f"led_{i}"] = state

    for i, confidence in enumerate(result.confidences, start=1):
        row[f"conf_{i}"] = round(float(confidence), 4)

    for i, detection in enumerate(detections, start=1):
        row[f"led_{i}_class_id"] = detection.get("class_id")
        row[f"led_{i}_class_name"] = detection.get("class_name")
        row[f"led_{i}_x1"] = round(float(detection.get("x1", 0.0)), 2)
        row[f"led_{i}_y1"] = round(float(detection.get("y1", 0.0)), 2)
        row[f"led_{i}_x2"] = round(float(detection.get("x2", 0.0)), 2)
        row[f"led_{i}_y2"] = round(float(detection.get("y2", 0.0)), 2)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Testet YOLODetector auf Frame(s).")
    parser.add_argument("--input", required=True, help="Pfad zu einem einzelnen Frame oder einem Ordner mit Frames")
    parser.add_argument("--config", default="configs/yolo_config.yaml", help="Pfad zur YOLO-Konfiguration")
    parser.add_argument("--debug-dir", default="data/debug/yolo", help="Ordner für Debug-Bilder mit Bounding Boxes")
    parser.add_argument("--output-csv", default="results/development_runs/yolo_results.csv", help="Pfad zur Ergebnis-CSV")
    parser.add_argument(
        "--no-debug-images",
        action="store_true",
        help="Wenn gesetzt, werden keine Debug-Bilder gespeichert.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    detector = YOLODetector(config)

    rows = []
    for frame_path in iter_frames(Path(args.input)):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"WARN: übersprungen (nicht lesbar): {frame_path}")
            continue

        result = detector.detect(frame)
        detections = result.debug_info.get("ordered_detections", [])
        print(f"{frame_path.name}: {result.led_state} detections={len(detections)} ({result.processing_time_ms:.2f} ms)")

        if not args.no_debug_images:
            save_debug_image(
                frame=frame,
                result=result,
                output_path=Path(args.debug_dir) / frame_path.name,
            )

        rows.append(build_result_row(frame_path, result))

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Ergebnisse gespeichert: {out}")

    if not args.no_debug_images:
        print(f"Debug-Bilder gespeichert: {Path(args.debug_dir)}")


if __name__ == "__main__":
    main()
