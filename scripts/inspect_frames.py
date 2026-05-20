from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.utils.config_loader import load_yaml_config
from src.utils.image_debug import save_debug_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Zeichnet LED-ROIs in ein einzelnes Frame.")
    parser.add_argument("--frame", required=True)
    parser.add_argument("--layout", default="configs/led_layout.yaml")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        raise FileNotFoundError(f"Frame konnte nicht geladen werden: {args.frame}")

    rois = load_yaml_config(args.layout)["leds"]
    for name, roi in rois.items():
        x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if args.output:
        save_debug_image(args.output, frame)
        print(f"Inspektionsbild gespeichert: {args.output}")
    else:
        temp_out = Path("data/debug/classic_cv/inspect_preview.jpg")
        save_debug_image(temp_out, frame)
        print(f"Kein --output gesetzt. Vorschau gespeichert unter {temp_out}")


if __name__ == "__main__":
    main()
