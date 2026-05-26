from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LEDRegion:
    led_id: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    source: str


@dataclass
class LocatorResult:
    regions: list[LEDRegion]
    status: str
    confidence: float
    debug_info: dict[str, Any] = field(default_factory=dict)


class BaseLEDLocator(ABC):
    @abstractmethod
    def locate(self, frame: np.ndarray) -> LocatorResult:
        raise NotImplementedError
