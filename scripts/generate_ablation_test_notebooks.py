"""Generate the six parameterized Kaggle test notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/ablation_test"
VARIANTS = (
    ("A0_YOLOv8n_Test.ipynb", "a0_yolov8n", "A0 YOLOv8n"),
    ("A1_P2_Test.ipynb", "a1_p2", "A1 P2"),
    ("A2_Shallow_Heavy_Test.ipynb", "a2_shallow_heavy", "A2 Shallow-Heavy"),
    ("A3_Terminal_P2_Test.ipynb", "a3_terminal_p2", "A3 Terminal P2"),
    ("A4_No_D5_Test.ipynb", "a4_no_d5", "A4 No-D5"),
    ("A5_TAD_YOLO_Test.ipynb", "a5_tad_yolo", "A5 TAD-YOLO"),
)


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def notebook(variant: str, title: str) -> dict:
    setup = f'''from pathlib import Path
import importlib.metadata as md
import subprocess, sys

VARIANT = "{variant}"
CHECKPOINT = None  # Set an explicit /kaggle/input/.../best.pt only if auto-discovery is ambiguous.
BATCH, IMGSZ, WORKERS, MAX_DET = 8, 640, 4, 300
REPO_URL, REPO_REF = "https://github.com/DongTienDinh/TAD-YOLO.git", "kaggle-ablation"
REPO = Path("/kaggle/working/TAD-YOLO")
PROJECT = Path("/kaggle/working/runs/ablation_test")
RUN_NAME = f"{{VARIANT}}_test"

if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(REPO)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)
required = {{"torch": "2.10.0", "torchvision": "0.25.0", "ultralytics-opencv-headless": "8.4.154", "PyYAML": "6.0.3"}}
installed = {{dist.metadata["Name"].lower(): dist.version for dist in md.distributions()}}
missing = [f"{{name}}=={{version}}" for name, version in required.items() if installed.get(name.lower(), "").split('+')[0] != version]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *missing], check=True)
print(f"Repository ready at {{REPO}}")
'''
    preflight = '''sys.path.insert(0, str(REPO))
import torch, ultralytics
from tad_yolo.offline import (
    find_ablation_checkpoint,
    resolve_visdrone_dataset,
    validate_dataset_split,
)

assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator"
assert ultralytics.__version__ == "8.4.154"
DATASET_YAML = resolve_visdrone_dataset()
TEST_SPLIT = validate_dataset_split(DATASET_YAML, "test", require_labels=True)
WEIGHTS = Path(CHECKPOINT).expanduser() if CHECKPOINT else find_ablation_checkpoint(VARIANT)
assert WEIGHTS.is_file(), WEIGHTS
print("GPU:", torch.cuda.get_device_name(0))
print("Checkpoint:", WEIGHTS)
print("Test split:", TEST_SPLIT)
'''
    evaluate = '''cmd = [
    sys.executable, str(REPO / "scripts/evaluate_ablation.py"),
    "--variant", VARIANT,
    "--weights", str(WEIGHTS),
    "--dataset", str(DATASET_YAML),
    "--split", "test",
    "--device", "0",
    "--imgsz", str(IMGSZ),
    "--batch", str(BATCH),
    "--workers", str(WORKERS),
    "--max-det", str(MAX_DET),
    "--project", str(PROJECT),
    "--name", RUN_NAME,
]
subprocess.run(cmd, check=True)
'''
    archive = '''import shutil

candidates = [p for p in PROJECT.glob(f"{RUN_NAME}*") if (p / "test_manifest.json").is_file()]
assert candidates, f"No completed test run found under {PROJECT}"
run_dir = max(candidates, key=lambda path: (path / "test_manifest.json").stat().st_mtime)
archive = shutil.make_archive(str(Path("/kaggle/working") / RUN_NAME), "zip", run_dir)
print("Test manifest:", run_dir / "test_manifest.json")
print("Archive:", archive)
'''
    return {
        "cells": [
            markdown(
                f"# {title} — VisDrone test evaluation\n\n"
                "Evaluation only: this notebook does not train. Enable one Kaggle T4 with Internet On, "
                "then attach (1) the labeled VisDrone dataset and (2) the saved output of the matching "
                "training notebook containing `best.pt`. Validation selected the checkpoint; this notebook "
                "uses the held-out `test` split exactly once.\n"
            ),
            code(setup),
            code(preflight),
            code(evaluate),
            code(archive),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, variant, title in VARIANTS:
        path = OUTPUT / filename
        path.write_text(json.dumps(notebook(variant, title), indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
