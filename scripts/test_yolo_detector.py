from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


def draw_debug_image(frame: Any, result: DetectionResult) -> Any:
    import cv2

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
    import cv2

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    debug_image = draw_debug_image(frame, result)
    if not cv2.imwrite(str(output), debug_image):
        raise OSError(f"Debug-Bild konnte nicht geschrieben werden: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Testet den YOLO-Detektor für ein einzelnes Frame.")
    parser.add_argument("--frame", required=True, help="Pfad zu einem einzelnen Eingabeframe")
    parser.add_argument("--config", default="configs/yolo_config.yaml", help="Pfad zur YOLO-Konfiguration")
    parser.add_argument("--output", help="Optionaler Pfad für ein Debug-Bild mit Bounding Boxes")
    args = parser.parse_args()

    import cv2

    frame_path = Path(args.frame)
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(f"Frame nicht lesbar: {frame_path}")

    config = load_yaml_config(args.config)
    detector = YOLODetector(config)
    result = detector.detect(frame)

    detections = result.debug_info.get("ordered_detections", [])
    print(f"YOLO LED state: {result.led_state}")
    print(f"Confidences: {[round(confidence, 4) for confidence in result.confidences]}")
    print(f"Detections: {len(detections)}")
    print("Bounding boxes:")
    for index, detection in enumerate(detections, start=1):
        print(
            "  "
            f"LED {index}: "
            f"class_id={detection['class_id']} "
            f"class_name={detection['class_name']} "
            f"confidence={float(detection['confidence']):.4f} "
            f"bbox=({float(detection['x1']):.1f}, {float(detection['y1']):.1f}, "
            f"{float(detection['x2']):.1f}, {float(detection['y2']):.1f})"
        )

    if args.output:
        save_debug_image(frame, result, args.output)
        print(f"Debug image saved: {args.output}")


if __name__ == "__main__":
    main()
