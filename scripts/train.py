"""Train the fixed TAD-YOLO F4 architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tad_yolo.offline import configure_offline_environment, resolve_visdrone_dataset

# This must run before importing Torch/Ultralytics on Kaggle.
configure_offline_environment()

import torch
import yaml

from tad_yolo.config import load_config
from tad_yolo.engine.trainer import TADYOLOTrainer
from tad_yolo.models.detector import TADYOLODetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/tad_yolov8n_f4.yaml",
    )
    parser.add_argument("--dataset", type=Path, help="YOLO-format dataset YAML")
    parser.add_argument("--device", help="One GPU index or 'cpu'")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--init", type=Path, help="Start a new run from a TAD-YOLO checkpoint")
    checkpoints.add_argument("--resume", type=Path, help="Resume an unfinished TAD-YOLO run")
    default_project = (
        Path("/kaggle/working/runs/tad-yolo")
        if Path("/kaggle/working").is_dir()
        else PROJECT_ROOT / "runs/train"
    )
    parser.add_argument("--project", type=Path, default=default_project)
    parser.add_argument("--name", default="tad_yolov8n_f4")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run shape, loss and backward checks")
    args = parser.parse_args()
    if not args.dry_run and args.dataset is None and not Path("/kaggle/input").is_dir():
        parser.error("--dataset is required unless --dry-run is used")
    if args.device and "," in args.device:
        parser.error("This runner supports one device per process")
    return args


def run_smoke_test(model: TADYOLODetector, device_name: str) -> None:
    device = torch.device("cpu" if device_name == "cpu" else f"cuda:{device_name}")
    model.to(device).eval()
    with torch.no_grad():
        features = model.forward_features(torch.randn(1, 3, 64, 96, device=device))
    expected = ((48, 16, 24), (64, 8, 12), (96, 4, 6))
    actual = tuple((x.shape[1], x.shape[2], x.shape[3]) for x in features)
    if actual != expected:
        raise AssertionError(f"Feature contract mismatch: {actual} != {expected}")

    model.train()
    batch = {
        "img": torch.rand(1, 3, 64, 64, device=device),
        "batch_idx": torch.tensor([0], device=device),
        "cls": torch.tensor([[0.0]], device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
    }
    loss, _ = model(batch)
    if not torch.isfinite(loss).all():
        raise AssertionError("Non-finite training loss")
    loss.sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("Missing or non-finite gradients")
    print(
        "PASS: shapes/loss/backward; "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}"
    )


def build_overrides(config: dict, args: argparse.Namespace, dataset_yaml: Path) -> dict:
    train = dict(config["train"])
    runtime = config["runtime"]
    precision = args.precision or runtime["precision"]
    overrides = {
        **train,
        "task": "detect",
        "mode": "train",
        "model": "yolov8n.yaml",
        "data": str(dataset_yaml),
        "imgsz": args.imgsz or config["model"]["input_size"],
        "device": args.device or runtime["device"],
        "workers": runtime["workers"] if args.workers is None else args.workers,
        "seed": runtime["seed"] if args.seed is None else args.seed,
        "deterministic": runtime["deterministic"],
        "amp": {"fp32": False, "fp16": True, "bf16": "bf16"}[precision],
        "compile": runtime["compile"],
        "project": str(args.project.resolve()),
        "name": args.name,
        "save": True,
        "save_period": 10,
        "cache": False,
        "pretrained": str(args.init.resolve()) if args.init else False,
        "resume": str(args.resume.resolve()) if args.resume else False,
        "val": True,
        "plots": not args.no_plots,
        "exist_ok": False,
    }
    for key in ("epochs", "batch"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    return overrides


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = config["runtime"]
    seed = runtime["seed"] if args.seed is None else args.seed
    torch.manual_seed(seed)

    model = TADYOLODetector.from_config(config)
    if args.dry_run:
        torch.set_num_threads(4)
        run_smoke_test(model, args.device or "cpu")
        return

    for path in (args.init, args.resume):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    dataset_yaml = resolve_visdrone_dataset(args.dataset)

    torch.backends.cuda.matmul.allow_tf32 = runtime["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["allow_tf32"]
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]

    trainer = TADYOLOTrainer(config, overrides=build_overrides(config, args, dataset_yaml))
    save_dir = Path(trainer.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "experiment_config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    contract = {
        "model": "TAD-YOLO-F4",
        "outputs": {"D2": 4, "D3": 8, "D4": 16},
        "terminal_p2": True,
        "has_d5": False,
        "seed": seed,
        "dataset": str(dataset_yaml),
    }
    (save_dir / "architecture_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    trainer.train()


if __name__ == "__main__":
    main()
