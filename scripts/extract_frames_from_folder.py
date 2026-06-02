from __future__ import annotations

import argparse
from pathlib import Path

from src.data.frame_extractor import extract_every_nth_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrahiert jedes n-te Frame aus allen Videos in einem Ordner."
    )
    parser.add_argument("--input-dir", required=True, help="Ordner mit Videos")
    parser.add_argument("--output-dir", required=True, help="Zielordner für extrahierte Frames")
    parser.add_argument("--step", type=int, default=30, help="Jedes n-te Frame speichern")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mp4", ".mov", ".avi", ".mkv"],
        help="Erlaubte Video-Dateiendungen",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input-Ordner nicht gefunden: {input_dir}")

    video_files = [
        file for file in input_dir.iterdir()
        if file.is_file() and file.suffix.lower() in args.extensions
    ]

    if not video_files:
        print(f"Keine Videos gefunden in: {input_dir}")
        return

    total_frames = 0

    for video_path in sorted(video_files):
        video_name = video_path.stem
        video_output_dir = output_dir / video_name

        print(f"Verarbeite: {video_path.name}")
        count = extract_every_nth_frame(
            str(video_path),
            str(video_output_dir),
            args.step
        )

        total_frames += count
        print(f"  Gespeicherte Frames: {count}")

    print("Fertig.")
    print(f"Verarbeitete Videos: {len(video_files)}")
    print(f"Gespeicherte Frames insgesamt: {total_frames}")


if __name__ == "__main__":
    main()