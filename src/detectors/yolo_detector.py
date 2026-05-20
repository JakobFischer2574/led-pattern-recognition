from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.detectors.base_detector import BaseDetector, DetectionResult
from src.yolo.yolo_to_led_array import yolo_boxes_to_led_array


class YOLODetector(BaseDetector):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_path = Path(config.get("model_path", "")) if config.get("model_path") else None
        self.model = None
        if self.model_path and self.model_path.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
            except ImportError as exc:
                raise ImportError("ultralytics ist nicht installiert. Installiere requirements oder entferne YOLO-Nutzung.") from exc
        elif self.model_path:
            raise FileNotFoundError(f"YOLO-Modellpfad nicht gefunden: {self.model_path}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self.model is None:
            raise RuntimeError("Kein YOLO-Modell geladen. Setze model_path in configs/yolo_config.yaml.")
        start = time.perf_counter()
        preds = self.model.predict(frame, conf=float(self.config.get("confidence_threshold", 0.25)), verbose=False)
        detections = []
        # TODO: Parsing der ultralytics-Ausgabe präzisieren
        led_state = yolo_boxes_to_led_array(detections)
        dt_ms = (time.perf_counter() - start) * 1000
        return DetectionResult(led_state=led_state, confidences=[0.0] * 5, processing_time_ms=dt_ms, debug_info={"raw_predictions": str(preds)})
