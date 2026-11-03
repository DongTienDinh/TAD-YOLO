"""Run image/video inference from a trained TAD-YOLO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tad_yolo.offline import configure_offline_environment

configure_offline_environment()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Image, directory, video or stream")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs/predict")
    parser.add_argument("--name", default="predict")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"File does not exist: {args.weights}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()), task="detect")
    model.predict(
        source=args.source,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project=str(args.project.resolve()),
        name=args.name,
        save=True,
    )


if __name__ == "__main__":
    main()
