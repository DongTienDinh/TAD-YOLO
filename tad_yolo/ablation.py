"""Ultralytics model factory for the cumulative TAD-YOLO ablation study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AblationVariant:
    key: str
    title: str
    config: Path
    expected_params: int
    expected_strides: tuple[int, ...]
    uses_dysample: bool = False


VARIANTS = {
    variant.key: variant
    for variant in (
        AblationVariant(
            "a0_yolov8n",
            "A0 - YOLOv8n baseline",
            PROJECT_ROOT / "configs/ablations/a0_yolov8n.yaml",
            3_012_798,
            (8, 16, 32),
        ),
        AblationVariant(
            "a1_p2",
            "A1 - YOLOv8n + P2",
            PROJECT_ROOT / "configs/ablations/a1_p2.yaml",
            2_927_880,
            (4, 8, 16, 32),
        ),
        AblationVariant(
            "a2_shallow_heavy",
            "A2 - P2 + shallow-heavy",
            PROJECT_ROOT / "configs/ablations/a2_shallow_heavy.yaml",
            1_633_136,
            (4, 8, 16, 32),
        ),
        AblationVariant(
            "a3_terminal_p2",
            "A3 - shallow-heavy + terminal P2",
            PROJECT_ROOT / "configs/ablations/a3_terminal_p2.yaml",
            1_633_552,
            (4, 8, 16, 32),
        ),
        AblationVariant(
            "a4_no_d5",
            "A4 - terminal P2 without D5",
            PROJECT_ROOT / "configs/ablations/a4_no_d5.yaml",
            1_199_142,
            (4, 8, 16),
        ),
        AblationVariant(
            "a5_tad_yolo",
            "A5 - no-D5 + DySample (TAD-YOLO)",
            PROJECT_ROOT / "configs/ablations/a5_tad_yolo.yaml",
            1_204_326,
            (4, 8, 16),
            uses_dysample=True,
        ),
    )
}


def register_custom_modules() -> None:
    """Expose DySample to Ultralytics' YAML parser before model construction."""
    import ultralytics.nn.tasks as tasks

    from tad_yolo.nn.upsampling import DySample

    tasks.DySample = DySample


def build_variant(key: str):
    """Build one randomly initialized ablation model from its frozen YAML."""
    if key not in VARIANTS:
        raise KeyError(f"Unknown variant {key!r}; choose from {sorted(VARIANTS)}")
    register_custom_modules()
    from ultralytics import YOLO

    variant = VARIANTS[key]
    if not variant.config.is_file():
        raise FileNotFoundError(variant.config)
    model = YOLO(str(variant.config), task="detect")
    params = sum(parameter.numel() for parameter in model.model.parameters())
    strides = tuple(int(value) for value in model.model.stride)
    if params != variant.expected_params:
        raise RuntimeError(
            f"{key} parameter contract changed: {params:,} != {variant.expected_params:,}"
        )
    if strides != variant.expected_strides:
        raise RuntimeError(
            f"{key} stride contract changed: {strides} != {variant.expected_strides}"
        )
    return model
