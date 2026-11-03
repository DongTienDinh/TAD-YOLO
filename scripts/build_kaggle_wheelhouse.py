"""Download a complete Python 3.12 Linux wheelhouse for offline Kaggle use.

Run this script once on a machine with Internet access. The produced ZIP is
uploaded to Kaggle as a Dataset/Input; it is never built inside an offline run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=PROJECT_ROOT / "requirements-kaggle-offline.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "kaggle_upload/tad-yolo-wheelhouse/wheelhouse.zip",
    )
    parser.add_argument("--python-version", default="312")
    parser.add_argument("--platform", default="manylinux_2_28_x86_64")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    requirements = args.requirements.resolve()
    if not requirements.is_file():
        raise FileNotFoundError(requirements)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tad_yolo_wheels_") as temporary:
        wheelhouse = Path(temporary) / "wheelhouse"
        wheelhouse.mkdir()
        command = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--dest",
            str(wheelhouse),
            "--only-binary=:all:",
            "--platform",
            args.platform,
            "--python-version",
            args.python_version,
            "--implementation",
            "cp",
            "--abi",
            f"cp{args.python_version}",
            "--requirement",
            str(requirements),
        ]
        print("+", " ".join(command))
        subprocess.run(command, check=True)

        shutil.copy2(requirements, wheelhouse / "_requirements.txt")
        wheels = sorted(wheelhouse.glob("*.whl"))
        if not wheels:
            raise RuntimeError("pip download produced no wheels")
        checksum_lines = [f"{sha256(path)}  {path.name}" for path in wheels]
        (wheelhouse / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        metadata = {
            "format": 1,
            "target": {
                "platform": args.platform,
                "python_version": args.python_version,
                "implementation": "cp",
                "abi": f"cp{args.python_version}",
            },
            "requirements_sha256": sha256(requirements),
            "wheel_count": len(wheels),
        }
        (wheelhouse / "METADATA.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(wheelhouse.iterdir()):
                archive.write(path, path.name)

    print(f"Wheelhouse: {output}")
    print(f"SHA256: {sha256(output)}")
    print("Upload this ZIP as a private Kaggle Dataset/Input.")


if __name__ == "__main__":
    main()
