"""Install the pinned TAD-YOLO environment without Kaggle Internet access.

This file intentionally uses only the Python standard library until the
offline wheelhouse has been installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "ultralytics": "8.4.154",
    "yaml": "6.0.3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--python-archive", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--force-reinstall", action="store_true")
    parser.add_argument("--keep-extracted-wheels", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def scan_input(filename: str, hints: tuple[str, ...]) -> Path:
    root = Path("/kaggle/input")
    if not root.is_dir():
        raise FileNotFoundError("/kaggle/input is unavailable")
    matches: list[tuple[int, Path]] = []
    for directory, subdirs, names in os.walk(root):
        relative = Path(directory).relative_to(root)
        if len(relative.parts) >= 4:
            subdirs.clear()
            continue
        if filename in names:
            candidate = Path(directory) / filename
            score = sum(hint in str(candidate).lower() for hint in hints)
            matches.append((score, candidate))
    if not matches:
        raise FileNotFoundError(f"Required Kaggle input not found: {filename}")
    matches.sort(key=lambda item: (-item[0], len(str(item[1]))))
    selected = matches[0][1]
    print(f"[input] {filename}: {selected}")
    return selected


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe tar member: {member.name}")
            if member.issym() or member.islnk():
                link = Path(member.linkname)
                if link.is_absolute() or ".." in link.parts:
                    raise ValueError(f"Unsafe tar link: {member.name} -> {member.linkname}")
        handle.extractall(destination)


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        handle.extractall(destination)


def find_python(root: Path) -> Path | None:
    candidates = sorted(root.glob("*/bin/python3.12")) + sorted(root.glob("*/bin/python"))
    return next((path for path in candidates if path.is_file()), None)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_wheelhouse(directory: Path) -> bool:
    checksum_file = directory / "SHA256SUMS"
    if not checksum_file.is_file():
        print("[warn] legacy wheelhouse has no SHA256SUMS; package versions will still be verified")
        return False
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split(maxsplit=1)
        path = directory / filename.strip()
        if not path.is_file():
            raise FileNotFoundError(f"Missing wheel: {filename}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Wheel checksum mismatch for {filename}")
    print("[verify] wheelhouse checksums passed")
    return True


def installed_versions(python: Path) -> dict[str, str] | None:
    code = (
        "import json,torch,torchvision,ultralytics,yaml;"
        "print(json.dumps({'torch':torch.__version__.split('+')[0],"
        "'torchvision':torchvision.__version__.split('+')[0],"
        "'ultralytics':ultralytics.__version__,'yaml':yaml.__version__}))"
    )
    result = subprocess.run([str(python), "-c", code], text=True, capture_output=True)
    if result.returncode:
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None


def write_launchers(working: Path, python: Path) -> None:
    environment = working / "tad_yolo_env.sh"
    launcher = working / "run_tad_yolo.sh"
    environment.write_text(
        "\n".join(
            (
                "export YOLO_OFFLINE=true",
                f"export YOLO_CONFIG_DIR={working}/.config",
                f"export MPLCONFIGDIR={working}/.matplotlib",
                "export MPLBACKEND=Agg",
                "export PIP_NO_INDEX=1",
                "export PIP_DISABLE_PIP_VERSION_CHECK=1",
                "export WANDB_MODE=disabled",
                "export COMET_MODE=DISABLED",
                "export HF_HUB_OFFLINE=1",
                "export TRANSFORMERS_OFFLINE=1",
                f"export PYTHONPATH={PROJECT_ROOT}",
                "",
            )
        ),
        encoding="utf-8",
    )
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"source {environment}\n"
        f"cd {PROJECT_ROOT}\n"
        f"exec {python} scripts/train.py --config configs/tad_yolov8n_f4.yaml --no-plots \"$@\"\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    print(f"[ready] environment: {environment}")
    print(f"[ready] train launcher: {launcher}")


def main() -> None:
    args = parse_args()
    working = args.working_dir.resolve()
    working.mkdir(parents=True, exist_ok=True)
    python_archive = (args.python_archive or scan_input(
        "cpython-312.tar.gz", ("tad-yolo", "ace-net", "cpython", "312")
    )).resolve()
    wheelhouse_archive = (args.wheelhouse or scan_input(
        "wheelhouse.zip", ("tad-yolo", "ace-net", "wheelhouse")
    )).resolve()

    runtime = working / "tad-python-3.12"
    python = find_python(runtime)
    if python is None:
        print(f"[extract] CPython -> {runtime}")
        safe_extract_tar(python_archive, runtime)
        python = find_python(runtime)
    if python is None:
        raise FileNotFoundError("The CPython archive has no bin/python3.12")
    version = subprocess.check_output(
        [str(python), "-c", "import sys;print('.'.join(map(str,sys.version_info[:3])))"],
        text=True,
    ).strip()
    if not version.startswith("3.12."):
        raise RuntimeError(f"Expected Python 3.12.x, got {version}")
    print(f"[python] {python} ({version})")

    current = installed_versions(python)
    if args.force_reinstall or current != EXPECTED:
        wheelhouse = working / "tad-wheelhouse"
        if wheelhouse.exists():
            shutil.rmtree(wheelhouse)
        print(f"[extract] wheelhouse -> {wheelhouse}")
        safe_extract_zip(wheelhouse_archive, wheelhouse)
        verify_wheelhouse(wheelhouse)
        requirements = wheelhouse / "_requirements.txt"
        if not requirements.is_file():
            requirements = wheelhouse / "_requirements_sanitized.txt"
        if not requirements.is_file():
            raise FileNotFoundError(
                "Wheelhouse has neither _requirements.txt nor _requirements_sanitized.txt"
            )
        command = [
            str(python), "-m", "pip", "install",
            "--no-index", "--no-cache-dir", "--only-binary=:all:",
            "--find-links", str(wheelhouse),
            "--requirement", str(requirements),
        ]
        print("+", " ".join(command))
        subprocess.run(
            command,
            check=True,
            env={**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        )
        current = installed_versions(python)
        if current != EXPECTED:
            raise RuntimeError(f"Installed versions do not match: {current} != {EXPECTED}")
        if not args.keep_extracted_wheels:
            shutil.rmtree(wheelhouse)
    else:
        print(f"[install] exact pinned environment already present: {current}")

    write_launchers(working, python)
    if not args.skip_preflight:
        environment = {
            **os.environ,
            "YOLO_OFFLINE": "true",
            "YOLO_CONFIG_DIR": str(working / ".config"),
            "MPLCONFIGDIR": str(working / ".matplotlib"),
            "MPLBACKEND": "Agg",
            "PIP_NO_INDEX": "1",
            "WANDB_MODE": "disabled",
            "COMET_MODE": "DISABLED",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": str(PROJECT_ROOT),
        }
        subprocess.run(
            [str(python), str(PROJECT_ROOT / "scripts/check_kaggle_environment.py"), "--strict"],
            check=True,
            env=environment,
        )
    print(f"Ready: {working / 'run_tad_yolo.sh'} --device 0 --epochs 150 --batch 16")


if __name__ == "__main__":
    main()
