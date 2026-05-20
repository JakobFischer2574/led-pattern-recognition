from __future__ import annotations

from pathlib import Path

import cv2


class VideoLoader:
    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video nicht gefunden: {self.video_path}")

    def open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Video kann nicht geöffnet werden: {self.video_path}")
        return cap
