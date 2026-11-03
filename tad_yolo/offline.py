"""Kaggle input discovery and network-free runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

import yaml


VISDRONE_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}


def find_kaggle_input(patterns: tuple[str, ...], *, filename: str | None = None) -> Path | None:
    """Find a Kaggle input by normalized name without walking image trees."""
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return None
    normalized = tuple(pattern.lower().replace("-", "_") for pattern in patterns)
    for directory, subdirs, files in os.walk(input_root):
        relative = Path(directory).relative_to(input_root)
        if len(relative.parts) >= 4:
            subdirs.clear()
            continue
        subdirs[:] = [
            name for name in subdirs
            if name.lower() not in {"images", "labels", "annotations", "__pycache__"}
        ]
        if filename and filename in files:
            return Path(directory) / filename
        for name in subdirs:
            clean_name = name.lower().replace("-", "_")
            if any(pattern == clean_name or pattern in clean_name for pattern in normalized):
                return Path(directory) / name
    return None


def configure_offline_environment(*, verbose: bool = False) -> dict[str, Any]:
    """Disable network integrations and configure writable Kaggle cache paths."""
    if not Path("/kaggle").exists():
        return {"kaggle": False, "offline": False}

    working = Path("/kaggle/working")
    environment = {
        "YOLO_OFFLINE": "true",
        "YOLO_CONFIG_DIR": str(working / ".config"),
        "MPLCONFIGDIR": str(working / ".matplotlib"),
        "MPLBACKEND": "Agg",
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "WANDB_MODE": "disabled",
        "COMET_MODE": "DISABLED",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    os.environ.update(environment)
    for path in (working / ".config", working / ".matplotlib"):
        path.mkdir(parents=True, exist_ok=True)

    assets = find_kaggle_input(
        ("tad-yolo-offline-assets", "ace-net-offline-assets", "offline-assets")
    )
    copied_fonts: list[str] = []
    if assets is not None:
        os.environ["TAD_YOLO_ASSETS"] = str(assets)
        font_target = working / ".config" / "Ultralytics"
        font_target.mkdir(parents=True, exist_ok=True)
        for name in ("Arial.ttf", "Arial.Unicode.ttf"):
            source = assets / "fonts" / name
            if source.is_file():
                shutil.copy2(source, font_target / name)
                copied_fonts.append(name)

    try:
        from ultralytics import settings

        updates: dict[str, Any] = {
            "sync": False,
            "clearml": False,
            "comet": False,
            "dvc": False,
            "mlflow": False,
            "raytune": False,
            "wandb": False,
        }
        if assets is not None and (assets / "weights").is_dir():
            updates["weights_dir"] = str((assets / "weights").resolve())
        settings.update(updates)
    except ImportError:
        # Expected before bootstrap installs the wheelhouse.
        pass

    status = {
        "kaggle": True,
        "offline": True,
        "assets": str(assets) if assets else None,
        "fonts": copied_fonts,
        **environment,
    }
    if verbose:
        print("TAD-YOLO Kaggle offline environment configured")
        print(f"  assets: {status['assets']}")
        print(f"  fonts: {copied_fonts or 'none'}")
        print(f"  config: {environment['YOLO_CONFIG_DIR']}")
    return status


def validate_dataset_yaml(path: str | Path, *, expected_classes: int = 10) -> dict[str, Any]:
    """Resolve train/val paths and validate the minimum YOLO dataset contract."""
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if not isinstance(names, (dict, list)) or len(names) != expected_classes:
        raise ValueError(f"Dataset must define exactly {expected_classes} class names")
    root = Path(data.get("path", yaml_path.parent)).expanduser()
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    resolved: dict[str, Any] = {"yaml": str(yaml_path), "root": str(root), "classes": len(names)}
    for split in ("train", "val"):
        value = data.get(split)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Dataset YAML is missing {split!r}")
        split_path = Path(value).expanduser()
        if not split_path.is_absolute():
            split_path = root / split_path
        split_path = split_path.resolve()
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset {split} path does not exist: {split_path}")
        resolved[split] = str(split_path)
    return resolved


def resolve_visdrone_dataset(explicit_yaml: str | Path | None = None) -> Path:
    """Use an explicit YAML or generate one for a mounted Kaggle VisDrone input."""
    if explicit_yaml is not None:
        path = Path(explicit_yaml).expanduser().resolve()
        validate_dataset_yaml(path)
        return path
    if not Path("/kaggle/input").is_dir():
        raise FileNotFoundError("Pass --dataset outside Kaggle")

    dataset = find_kaggle_input(
        ("visdrone_det_yolo", "visdrone-yolo", "visdrone2019", "visdrone")
    )
    if dataset is None:
        raise FileNotFoundError("No VisDrone input found under /kaggle/input")

    candidates = {
        "train": (
            "VisDrone2019-DET-train/images",
            "images/train",
            "train/images",
        ),
        "val": (
            "VisDrone2019-DET-val/images",
            "images/val",
            "val/images",
        ),
        "test": (
            "VisDrone2019-DET-test-dev/images",
            "images/test",
            "test/images",
        ),
    }
    selected: dict[str, str] = {}
    for split, options in candidates.items():
        selected[split] = next((item for item in options if (dataset / item).exists()), "")
    if not selected["train"] or not selected["val"]:
        raise FileNotFoundError(
            f"VisDrone input found at {dataset}, but train/val image directories are unsupported"
        )
    selected["test"] = selected["test"] or selected["val"]

    runtime_yaml = Path("/kaggle/working/visdrone_tad_yolo.yaml")
    runtime_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset.resolve()),
                "train": selected["train"],
                "val": selected["val"],
                "test": selected["test"],
                "names": VISDRONE_NAMES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    validate_dataset_yaml(runtime_yaml)
    return runtime_yaml
