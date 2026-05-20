from __future__ import annotations

import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray, resize_width: int | None, blur_kernel_size: int) -> np.ndarray:
    output = frame.copy()
    if resize_width:
        h, w = output.shape[:2]
        scale = resize_width / float(w)
        output = cv2.resize(output, (resize_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    if blur_kernel_size and blur_kernel_size > 1:
        k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        output = cv2.GaussianBlur(output, (k, k), 0)
    return output
