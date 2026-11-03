"""Validate a TAD-YOLO checkpoint on a YOLO-format dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tad_yolo.offline import configure_offline_environment, resolve_visdrone_dataset

configure_offline_environment()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Auto-detected on Kaggle when omitted")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs/validation")
    parser.add_argument("--name", default="val")
    args = parser.parse_args()
    for path in (args.weights,):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    dataset_yaml = resolve_visdrone_dataset(args.dataset)

    import torch
    from ultralytics.models.yolo.detect import DetectionValidator

    if args.device == "cpu":
        torch.set_num_threads(4)
    validator = DetectionValidator(args={
        "model": str(args.weights.resolve()),
        "data": str(dataset_yaml),
        "split": args.split,
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "max_det": args.max_det,
        "project": str(args.project.resolve()),
        "name": args.name,
        "task": "detect",
        "mode": "val",
        "plots": False,
        "exist_ok": False,
    })
    metrics = validator(model=str(args.weights.resolve()))
    output = Path(validator.save_dir) / "evaluation.json"
    output.write_text(
        json.dumps(
            {
                "weights": str(args.weights.resolve()),
                "dataset": str(dataset_yaml),
                "split": args.split,
                "metrics": {key: float(value) for key, value in metrics.items()},
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Validation evidence: {output}")


if __name__ == "__main__":
    main()
