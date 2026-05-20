from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DetectionResult:
    led_state: list[int]
    confidences: list[float]
    processing_time_ms: float
    debug_info: dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        raise NotImplementedError
