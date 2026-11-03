"""Check the architecture contract, tensor shapes and parameter count."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from tad_yolo.config import load_config
from tad_yolo.models.detector import TADYOLODetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/tad_yolov8n_f4.yaml",
    )
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    if args.height % 32 or args.width % 32:
        parser.error("height and width must be divisible by 32")

    torch.set_num_threads(4)
    config = load_config(args.config)
    model = TADYOLODetector.from_config(config).eval()
    with torch.no_grad():
        features = model.forward_features(torch.zeros(1, 3, args.height, args.width))

    print(f"Model: {model.model_name}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    for name, feature, stride in zip(("D2/T2", "D3/T3", "D4/N4"), features, model.strides):
        print(f"{name}: shape={tuple(feature.shape)}, stride={int(stride)}")


if __name__ == "__main__":
    main()
