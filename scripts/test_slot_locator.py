from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from src.classic_cv.locators import SlotBasedLEDLocator, TrackingLEDLocator
from src.utils.config_loader import load_yaml_config
from src.utils.image_debug import draw_detection_debug_image, save_debug_image


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
    parser = argparse.ArgumentParser(description="Testet den SlotBasedLEDLocator auf Bilddateien.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/classic_cv_config.yaml")
    parser.add_argument("--output", default="data/debug/slot_locator")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    locator_cfg = cfg.get("locator", {})
    if locator_cfg.get("type", "slot_based") != "slot_based":
        raise ValueError("Für test_slot_locator.py muss locator.type=slot_based gesetzt sein.")

    slot = SlotBasedLEDLocator(locator_cfg)
    tracking_cfg = locator_cfg.get("tracking", {})
    locator = TrackingLEDLocator(
        slot,
        enabled=bool(tracking_cfg.get("enabled", True)),
        max_tracking_fallback_frames=int(tracking_cfg.get("max_tracking_fallback_frames", 5)),
        fallback_confidence=float(tracking_cfg.get("fallback_confidence", 0.5)),
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for frame_path in iter_frames(Path(args.input)):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"WARN: übersprungen (nicht lesbar): {frame_path}")
            continue
        result = locator.locate(frame)
        print(f"{frame_path.name}: status={result.status} conf={result.confidence:.3f} selected={len(result.regions)}")

        debug_metrics = []
        for region in sorted(result.regions, key=lambda r: r.led_id):
            debug_metrics.append({
                "led_id": region.led_id,
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height,
                "state": 0,
                "confidence": region.confidence,
            })

        debug_img = draw_detection_debug_image(
            frame=frame,
            debug_info=debug_metrics,
            frame_name=frame_path.name,
            layout_path="n/a",
            config_path=args.config,
            locator_data=result.debug_info,
            locator_type="slot_based",
            locator_status=result.status,
            locator_confidence=result.confidence,
        )
        save_debug_image(output_dir / frame_path.name, debug_img)

        row = {
            "frame_name": frame_path.name,
            "locator_status": result.status,
            "locator_confidence": round(float(result.confidence), 4),
            "selected_count": len(result.regions),
        }
        regions_sorted = sorted(result.regions, key=lambda r: r.led_id)
        for i in range(1, 6):
            if i <= len(regions_sorted):
                rr = regions_sorted[i - 1]
                row[f"led_{i}_x"] = rr.x
                row[f"led_{i}_y"] = rr.y
            else:
                row[f"led_{i}_x"] = ""
                row[f"led_{i}_y"] = ""
        rows.append(row)

    csv_path = output_dir / "slot_locator_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"CSV gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
