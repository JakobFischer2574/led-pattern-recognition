from __future__ import annotations

from pathlib import Path

import cv2

from src.data.video_loader import VideoLoader


def extract_every_nth_frame(video_path: str | Path, output_dir: str | Path, step: int = 30, image_format: str = "jpg") -> int:
    if step <= 0:
        raise ValueError("step muss > 0 sein")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = VideoLoader(video_path).open_capture()

    saved = 0
    frame_idx = 0
    stem = Path(video_path).stem
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            target = out / f"{stem}_frame_{frame_idx:06d}.{image_format}"
            cv2.imwrite(str(target), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    return saved
