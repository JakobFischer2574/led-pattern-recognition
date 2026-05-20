from __future__ import annotations

import argparse

from src.yolo.dataset_preparation import copy_annotation_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Kopiert Frames für spätere YOLO-Annotation.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/annotation_candidates")
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    copied = copy_annotation_candidates(args.input, args.output, args.step)
    print(f"Kopierte Frames für Annotation: {copied}")


if __name__ == "__main__":
    main()
