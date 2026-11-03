"""Build a checksum-protected self-extracting TAD-YOLO source bundle."""

from __future__ import annotations

import base64
import hashlib
import io
import lzma
import os
from pathlib import Path
import tarfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "kaggle_upload/tad-yolo-code/gen-full.py"
EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "__pycache__", ".pytest_cache", "runs", "kaggle_upload"
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".onnx", ".engine", ".zip"}


def collect_files() -> list[Path]:
    files: list[Path] = []
    for directory, subdirs, names in os.walk(PROJECT_ROOT):
        subdirs[:] = [
            name for name in subdirs
            if name not in EXCLUDED_DIRECTORIES
            and not name.startswith(".")
            and not name.endswith(".egg-info")
        ]
        for name in names:
            path = Path(directory) / name
            if name.startswith(".") or path.suffix in EXCLUDED_SUFFIXES:
                continue
            files.append(path)
    return sorted(files)


def main() -> None:
    files = collect_files()
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(PROJECT_ROOT)))
    compressed = lzma.compress(archive_buffer.getvalue(), preset=6)
    encoded = base64.b64encode(compressed).decode("ascii")
    checksum = hashlib.sha256(compressed).hexdigest()

    unpacker = f'''# Generated TAD-YOLO source restore script.
import base64
import hashlib
import io
import lzma
from pathlib import Path
import tarfile

FORMAT_VERSION = 1
DEFAULT_TARGET = "TAD-YOLO"
ENTRY_COUNT = {len(files)}
PAYLOAD_SHA256 = "{checksum}"
PAYLOAD = "{encoded}"


def restore(target: Path) -> int:
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    compressed = base64.b64decode(PAYLOAD.encode("ascii"))
    actual = hashlib.sha256(compressed).hexdigest()
    if actual != PAYLOAD_SHA256:
        raise ValueError(f"Source bundle checksum mismatch: {{actual}} != {{PAYLOAD_SHA256}}")
    count = 0
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(compressed)), mode="r:") as archive:
        for member in archive.getmembers():
            destination = (target / member.name).resolve()
            if destination != target and target not in destination.parents:
                raise ValueError(f"Unsafe archive member: {{member.name}}")
            if member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Unable to read {{member.name}}")
                destination.write_bytes(source.read())
                count += 1
    if count != ENTRY_COUNT:
        raise ValueError(f"Restored {{count}} entries; expected {{ENTRY_COUNT}}")
    return count


if __name__ == "__main__":
    target = Path("/kaggle/working") / DEFAULT_TARGET if Path("/kaggle/working").exists() else Path.cwd() / DEFAULT_TARGET
    count = restore(target)
    print(f"Restored {{count}} files into {{target.resolve()}}")
'''
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(unpacker, encoding="utf-8")
    print(f"Packed {len(files)} files into {OUTPUT}")
    print(f"Payload SHA256: {checksum}")


if __name__ == "__main__":
    main()
