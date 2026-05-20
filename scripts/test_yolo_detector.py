from __future__ import annotations

import argparse

import cv2

from src.detectors.yolo_detector import YOLODetector
from src.utils.config_loader import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Testet YOLO-Detektor-Gerüst.")
    parser.add_argument("--frame", required=True)
    parser.add_argument("--config", default="configs/yolo_config.yaml")
    args = parser.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        raise FileNotFoundError(f"Frame nicht lesbar: {args.frame}")

    detector = YOLODetector(load_yaml_config(args.config))
    result = detector.detect(frame)
    print(result)


if __name__ == "__main__":
    main()
