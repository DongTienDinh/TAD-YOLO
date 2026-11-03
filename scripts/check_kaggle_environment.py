"""Strict preflight for a fully offline TAD-YOLO Kaggle run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tad_yolo.offline import configure_offline_environment, resolve_visdrone_dataset


EXPECTED = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "ultralytics": "8.4.154",
    "yaml": "6.0.3",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    status = configure_offline_environment(verbose=True)

    if sys.version_info[:2] != (3, 12):
        failures.append(f"Python must be 3.12.x, got {sys.version.split()[0]}")
    for name in ("YOLO_OFFLINE", "PIP_NO_INDEX", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) not in {"1", "true"}:
            failures.append(f"{name} is not enabled")

    try:
        import torch
        import torchvision
        import ultralytics
        import yaml

        installed = {
            "torch": torch.__version__.split("+")[0],
            "torchvision": torchvision.__version__.split("+")[0],
            "ultralytics": ultralytics.__version__,
            "yaml": yaml.__version__,
        }
        print(f"Versions: {installed}")
        if installed != EXPECTED:
            failures.append(f"Version mismatch: {installed} != {EXPECTED}")
        if not torch.cuda.is_available():
            failures.append("CUDA is unavailable")
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {memory:.1f} GiB")
            bf16_supported = torch.cuda.is_bf16_supported()
            print(f"Native BF16: {bf16_supported}")
            if not bf16_supported:
                print("Precision recommendation: pass --precision fp16")
    except Exception as error:
        failures.append(f"Core dependency import failed: {error}")
        torch = None

    assets_value = status.get("assets")
    if not assets_value:
        failures.append("Offline assets input was not found")
    else:
        assets = Path(assets_value)
        for relative in ("fonts/Arial.ttf", "weights/yolo26n.pt"):
            if not (assets / relative).is_file():
                failures.append(f"Missing offline asset: {assets / relative}")

    try:
        dataset_yaml = resolve_visdrone_dataset()
        print(f"Dataset: {dataset_yaml}")
    except Exception as error:
        failures.append(f"Dataset preflight failed: {error}")

    if torch is not None:
        try:
            from tad_yolo.config import load_config
            from tad_yolo.models.detector import TADYOLODetector

            config = load_config(PROJECT_ROOT / "configs/tad_yolov8n_f4.yaml")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = TADYOLODetector.from_config(config).to(device).eval()
            with torch.no_grad():
                outputs = model.forward_features(torch.zeros(1, 3, 64, 64, device=device))
            shapes = [tuple(output.shape) for output in outputs]
            expected_shapes = [(1, 48, 16, 16), (1, 64, 8, 8), (1, 96, 4, 4)]
            print(f"Feature shapes: {shapes}")
            if shapes != expected_shapes:
                failures.append(f"Architecture shape mismatch: {shapes}")
        except Exception as error:
            failures.append(f"Architecture smoke test failed: {error}")

    if failures:
        print("Preflight failures:")
        for failure in failures:
            print(f"  - {failure}")
        if args.strict:
            raise SystemExit(2)
    else:
        print("PASS: Kaggle offline environment is ready")


if __name__ == "__main__":
    main()
