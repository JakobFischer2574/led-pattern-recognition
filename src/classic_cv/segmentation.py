from __future__ import annotations

import cv2
import numpy as np


def to_value_channel(roi_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2]
