"""Configuration loading and architecture-contract validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _validate_architecture_contract(config)
    return config


def _validate_architecture_contract(config: dict[str, Any]) -> None:
    required = {"model", "train", "runtime"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing top-level config sections: {sorted(missing)}")

    model = config["model"]
    expected = {
        "stem_channels": 24,
        "backbone_channels": [48, 64, 96, 128],
        "neck_channels": [48, 64, 96],
        "detection_strides": [4, 8, 16],
        "terminal_p2": True,
        "has_d5": False,
    }
    mismatches = {
        key: (model.get(key), value)
        for key, value in expected.items()
        if model.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Config no longer describes the fixed F4 architecture: {mismatches}")

    dysample = model.get("dysample", {})
    if (
        dysample.get("scale") != 2
        or dysample.get("groups") != 4
        or dysample.get("style") != "lp"
    ):
        raise ValueError("F4 requires DySample scale=2, groups=4 and style='lp'")
