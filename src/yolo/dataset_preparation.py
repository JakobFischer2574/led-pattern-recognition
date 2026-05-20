from __future__ import annotations

import shutil
from pathlib import Path


def copy_annotation_candidates(input_dir: str | Path, output_dir: str | Path, step: int = 1) -> int:
    src = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in src.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    copied = 0
    for i, file in enumerate(files):
        if i % step == 0:
            shutil.copy2(file, dst / file.name)
            copied += 1
    return copied
