"""Train one frozen TAD-YOLO ablation variant with a shared protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from tad_yolo.ablation import VARIANTS, build_variant
from tad_yolo.offline import resolve_visdrone_dataset, validate_dataset_yaml


PAPER_CONFIG = PROJECT_ROOT / "configs/tad_yolov8n_f4_paper300.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--dataset", type=Path, help="VisDrone YOLO YAML; auto-detected on Kaggle")
    parser.add_argument("--device", default="0", help="One CUDA index or 'cpu'")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--resume", type=Path, help="Resume from this variant's last.pt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    default_project = (
        Path("/kaggle/working/runs/ablation")
        if Path("/kaggle/working").is_dir()
        else PROJECT_ROOT / "runs/ablation"
    )
    parser.add_argument("--project", type=Path, default=default_project)
    parser.add_argument("--name")
    args = parser.parse_args()
    if "," in args.device:
        parser.error("The Kaggle notebooks are intentionally single-GPU; pass one device index")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"Checkpoint does not exist: {args.resume}")
    return args


def write_profile(args: argparse.Namespace, model, dataset_yaml: Path | None) -> Path:
    variant = VARIANTS[args.variant]
    profile_dir = args.project.resolve() / "_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / f"{args.variant}_seed{args.seed}.json"
    params = sum(p.numel() for p in model.model.parameters())
    gradients = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    layers = len(list(model.model.modules()))
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = get_flops(model.model, imgsz=args.imgsz)
    except Exception as error:
        gflops = None
        profile_error = str(error)
    else:
        profile_error = None
    payload = {
        "variant": args.variant,
        "title": variant.title,
        "model_yaml": str(variant.config),
        "dataset_yaml": str(dataset_yaml) if dataset_yaml else None,
        "seed": args.seed,
        "input_size": args.imgsz,
        "parameters": params,
        "strides": [int(value) for value in model.model.stride],
        "layers": layers,
        "gradients": gradients,
        "gflops": gflops,
        "gflops_note": "THOP approximation; grid_sample may be under-counted for A5",
        "profile_error": profile_error,
        "torch": torch.__version__,
    }
    profile_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return profile_path


def smoke_test(model, device_name: str) -> None:
    device = torch.device("cpu" if device_name == "cpu" else f"cuda:{device_name}")
    model.model.to(device).eval()
    with torch.no_grad():
        output = model.model(torch.zeros(1, 3, 64, 64, device=device))
    if output is None:
        raise RuntimeError("Model forward returned None")
    print("PASS: model construction and forward smoke test")


def main() -> None:
    args = parse_args()
    variant = VARIANTS[args.variant]
    print(f"Variant: {variant.title}")
    model = build_variant(args.variant)

    if args.dry_run:
        write_profile(args, model, None)
        smoke_test(model, args.device)
        return

    dataset_yaml = resolve_visdrone_dataset(args.dataset)
    dataset = validate_dataset_yaml(dataset_yaml)
    print(f"Dataset: {json.dumps(dataset, indent=2)}")

    if args.resume:
        from tad_yolo.ablation import register_custom_modules
        from ultralytics import YOLO

        register_custom_modules()
        model = YOLO(str(args.resume.resolve()))
        write_profile(args, model, dataset_yaml)
        model.train(resume=True, device=args.device, workers=args.workers)
        return

    config = yaml.safe_load(PAPER_CONFIG.read_text(encoding="utf-8"))
    train = dict(config["train"])
    if args.epochs is not None:
        train["epochs"] = args.epochs
    if args.batch is not None:
        train["batch"] = args.batch

    name = args.name or f"{args.variant}_seed{args.seed}"
    write_profile(args, model, dataset_yaml)
    results = model.train(
        **train,
        data=str(dataset_yaml),
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        amp={"fp32": False, "fp16": "fp16", "bf16": "bf16"}[args.precision],
        pretrained=False,
        patience=0,
        project=str(args.project.resolve()),
        name=name,
        save=True,
        save_period=10,
        cache=False,
        val=True,
        plots=not args.no_plots,
        exist_ok=False,
    )
    save_dir = Path(model.trainer.save_dir)
    manifest = {
        "variant": args.variant,
        "title": variant.title,
        "seed": args.seed,
        "dataset": str(dataset_yaml),
        "model_yaml": str(variant.config),
        "save_dir": str(save_dir),
        "best": str(save_dir / "weights/best.pt"),
        "last": str(save_dir / "weights/last.pt"),
        "results_type": type(results).__name__,
    }
    (save_dir / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
