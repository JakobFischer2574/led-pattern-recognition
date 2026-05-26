from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from src.detectors.classic_cv_detector import ClassicCVDetector
from src.utils.config_loader import load_yaml_config
from src.utils.image_debug import save_detection_debug_artifacts


def iter_frames(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input-Pfad nicht gefunden: {path}")
    if path.is_file():
        return [path]
    frames = sorted([p for p in path.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if not frames:
        raise ValueError(f"Keine Bilddateien in Ordner gefunden: {path}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Testet ClassicCVDetector auf Frame(s).")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/classic_cv_config.yaml")
    parser.add_argument("--layout", default="configs/led_layout.yaml")
    parser.add_argument("--debug-dir", default="data/debug/classic_cv")
    parser.add_argument("--output-csv", default="results/development_runs/classic_cv_results.csv")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    layout = load_yaml_config(args.layout).get("leds")
    if not isinstance(layout, dict):
        raise ValueError("Ungültiges Layout: 'leds' fehlt oder ist kein dict")

    detector = ClassicCVDetector(cfg, layout)

    rows = []
    for frame_path in iter_frames(Path(args.input)):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"WARN: übersprungen (nicht lesbar): {frame_path}")
            continue
        result = detector.detect(frame)
        print(f"{frame_path.name}: {result.led_state} locator={result.locator_status} ({result.processing_time_ms:.2f} ms)")
        if cfg.get("debug", {}).get("save_debug_images", False):
            dbg_cfg = cfg.get("debug", {})
            save_detection_debug_artifacts(
                output_dir=Path(args.debug_dir),
                frame=frame,
                frame_name=frame_path.name,
                layout_path=args.layout,
                config_path=args.config,
                debug_info=result.debug_info.get("metrics", []),
                save_roi_crops=bool(dbg_cfg.get("save_roi_crops", False)),
                roi_scale=int(dbg_cfg.get("roi_crop_scale", 4)),
                save_masks=bool(dbg_cfg.get("save_masks", False)),
                locator_data=result.debug_info.get("locator", {}),
                locator_type=result.debug_info.get("locator_type", "n/a"),
                locator_status=result.locator_status,
                locator_confidence=result.locator_confidence,
            )

        row = {
            "frame_name": frame_path.name,
            "mean_latency_ms": round(result.processing_time_ms, 3),
            "locator_status": result.locator_status,
            "locator_confidence": round(result.locator_confidence, 3),
        }
        for i, state in enumerate(result.led_state, start=1):
            row[f"led_{i}"] = state
        rows.append(row)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Ergebnisse gespeichert: {out}")


if __name__ == "__main__":
    main()
