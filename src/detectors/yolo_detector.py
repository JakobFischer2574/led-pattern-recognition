from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.detectors.base_detector import BaseDetector, DetectionResult
from src.yolo.yolo_to_led_array import yolo_boxes_to_led_array


class YOLODetector(BaseDetector):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.confidence_threshold = float(config.get("confidence_threshold", 0.25))
        self.device = str(config.get("device", "cpu"))
        self.expected_led_count = int(config.get("expected_led_count", 5))
        self.class_mapping = self._parse_class_mapping(config.get("class_mapping", {}))
        self.model_path = self._resolve_model_path(config.get("model_path"))

        if importlib.util.find_spec("ultralytics") is None:
            raise ImportError(
                "ultralytics ist nicht installiert. Installiere die Projekt-Dependencies "
                "oder füge ultralytics zur aktuellen Python-Umgebung hinzu."
            )

        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))

    @staticmethod
    def _parse_class_mapping(raw_mapping: Any) -> dict[int, str]:
        if not isinstance(raw_mapping, dict):
            return {}
        parsed: dict[int, str] = {}
        for raw_class_id, raw_class_name in raw_mapping.items():
            try:
                class_id = int(raw_class_id)
            except (TypeError, ValueError):
                continue
            parsed[class_id] = str(raw_class_name)
        return parsed

    @staticmethod
    def _resolve_model_path(raw_model_path: Any) -> Path:
        if raw_model_path is None or str(raw_model_path).strip() == "":
            raise ValueError("YOLO model_path ist leer. Setze model_path in configs/yolo_config.yaml.")

        model_path = Path(str(raw_model_path)).expanduser()
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        model_path = model_path.resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO-Modellpfad nicht gefunden: {model_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO-Modellpfad ist keine Datei: {model_path}")
        return model_path

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if frame is None or frame.size == 0:
            raise ValueError("Frame ist leer oder ungültig.")

        start = time.perf_counter()
        predictions = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        detections = self._parse_predictions(predictions)
        led_state, confidences, ordered_detections = yolo_boxes_to_led_array(
            detections,
            expected_led_count=self.expected_led_count,
        )
        dt_ms = (time.perf_counter() - start) * 1000

        return DetectionResult(
            led_state=led_state,
            confidences=confidences,
            processing_time_ms=dt_ms,
            locator_status="ok" if ordered_detections else "not_found",
            locator_confidence=max(confidences) if confidences else 0.0,
            debug_info={
                "detections": detections,
                "ordered_detections": ordered_detections,
                "raw_detection_count": len(detections),
                "ordered_detection_count": len(ordered_detections),
                "expected_led_count": self.expected_led_count,
                "model_path": str(self.model_path),
                "confidence_threshold": self.confidence_threshold,
                "device": self.device,
                "class_mapping": self.class_mapping,
            },
        )

    def _parse_predictions(self, predictions: Any) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        model_names = getattr(self.model, "names", {}) or {}

        for result in predictions:
            boxes = getattr(result, "boxes", None)
            result_names = getattr(result, "names", None) or model_names
            if boxes is None:
                continue

            for box in boxes:
                class_id = self._to_int(getattr(box, "cls", None))
                confidence = self._to_float(getattr(box, "conf", None))
                xyxy = self._to_float_list(getattr(box, "xyxy", None))
                if class_id is None or confidence is None or len(xyxy) < 4:
                    continue

                x1, y1, x2, y2 = xyxy[:4]
                class_name = self._class_name_for_id(class_id, result_names)
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "x_center": (x1 + x2) / 2.0,
                        "y_center": (y1 + y2) / 2.0,
                        "width": x2 - x1,
                        "height": y2 - y1,
                    }
                )

        return detections

    def _class_name_for_id(self, class_id: int, result_names: Any) -> str:
        if class_id in self.class_mapping:
            return self.class_mapping[class_id]
        if isinstance(result_names, dict) and class_id in result_names:
            return str(result_names[class_id])
        if isinstance(result_names, list) and 0 <= class_id < len(result_names):
            return str(result_names[class_id])
        return "unknown"

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()
            array = np.asarray(value).reshape(-1)
            if array.size == 0:
                return None
            return float(array[0])
        except (TypeError, ValueError):
            return None

    @classmethod
    def _to_int(cls, value: Any) -> int | None:
        float_value = cls._to_float(value)
        if float_value is None:
            return None
        return int(float_value)

    @staticmethod
    def _to_float_list(value: Any) -> list[float]:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()
            return [float(item) for item in np.asarray(value).reshape(-1).tolist()]
        except (TypeError, ValueError):
            return []
