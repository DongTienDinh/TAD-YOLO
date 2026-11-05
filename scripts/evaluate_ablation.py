"""Evaluate one trained ablation checkpoint on a labeled YOLO split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tad_yolo.ablation import VARIANTS, register_custom_modules
from tad_yolo.offline import (
    configure_offline_environment,
    resolve_visdrone_dataset,
    validate_dataset_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Auto-detected on Kaggle when omitted")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    default_project = (
        Path("/kaggle/working/runs/ablation_test")
        if Path("/kaggle/working").is_dir()
        else PROJECT_ROOT / "runs/ablation_test"
    )
    parser.add_argument("--project", type=Path, default=default_project)
    parser.add_argument("--name")
    args = parser.parse_args()
    if not args.weights.expanduser().is_file():
        parser.error(f"Checkpoint does not exist: {args.weights}")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def maximum_objects_per_image(labels_dir: str | Path) -> int:
    """Return the largest number of non-empty YOLO rows in one label file."""
    maximum = 0
    for label in Path(labels_dir).rglob("*.txt"):
        with label.open("r", encoding="utf-8") as stream:
            maximum = max(maximum, sum(bool(line.strip()) for line in stream))
    return maximum


def find_new_run_directory(project: Path, name: str, before: set[Path]) -> Path:
    """Locate the output directory created by an Ultralytics validation call."""
    candidates = [
        path.resolve()
        for path in project.glob(f"{name}*")
        if path.is_dir() and path.resolve() not in before
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)
    existing = [path.resolve() for path in project.glob(f"{name}*") if path.is_dir()]
    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime_ns)
    raise RuntimeError(f"Ultralytics created no output directory under {project} for {name!r}")


def main() -> None:
    args = parse_args()
    configure_offline_environment()
    weights = args.weights.expanduser().resolve()
    dataset_yaml = resolve_visdrone_dataset(args.dataset)
    split_info = validate_dataset_split(dataset_yaml, args.split, require_labels=True)

    register_custom_modules()
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    variant = VARIANTS[args.variant]
    params = sum(parameter.numel() for parameter in model.model.parameters())
    strides = tuple(int(value) for value in model.model.stride)
    if params != variant.expected_params or strides != variant.expected_strides:
        raise RuntimeError(
            f"Checkpoint does not match {args.variant}: params={params:,}, strides={strides}; "
            f"expected params={variant.expected_params:,}, strides={variant.expected_strides}"
        )

    run_name = args.name or f"{args.variant}_{args.split}"
    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in project.glob(f"{run_name}*") if path.is_dir()}
    observed_max_objects = maximum_objects_per_image(split_info["labels"])
    effective_max_det = max(args.max_det, observed_max_objects)
    metrics = model.val(
        data=str(dataset_yaml),
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        quantize=16 if args.device != "cpu" else None,
        conf=0.001,
        iou=0.7,
        max_det=effective_max_det,
        plots=not args.no_plots,
        save_json=False,
        project=str(project),
        name=run_name,
        exist_ok=False,
        verbose=True,
    )
    save_dir = find_new_run_directory(project, run_name, before)
    results = {
        key: float(value)
        for key, value in getattr(metrics, "results_dict", {}).items()
    }
    speed = {
        key: float(value)
        for key, value in getattr(metrics, "speed", {}).items()
    }
    manifest = {
        "variant": args.variant,
        "title": variant.title,
        "checkpoint": str(weights),
        "checkpoint_sha256": sha256(weights),
        "checkpoint_bytes": weights.stat().st_size,
        "dataset_yaml": str(dataset_yaml),
        "split": split_info,
        "parameters": params,
        "strides": list(strides),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": 0.001,
        "iou": 0.7,
        "requested_max_det": args.max_det,
        "observed_max_objects_per_image": observed_max_objects,
        "max_det": effective_max_det,
        "precision": "fp16" if args.device != "cpu" else "fp32",
        "metrics": results,
        "speed_ms_per_image": speed,
        "save_dir": str(save_dir),
    }
    output = save_dir / "test_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Test evidence: {output}")


if __name__ == "__main__":
    main()
