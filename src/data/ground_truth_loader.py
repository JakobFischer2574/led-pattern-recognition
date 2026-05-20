from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"video_filename", "error_code", "environment"}


def load_ground_truth(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-Truth CSV nicht gefunden: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Ground-Truth CSV hat fehlende Spalten: {sorted(missing)}")
    return df
