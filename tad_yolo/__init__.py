"""TAD-YOLO: a focused implementation of the F4 detector."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tad_yolo.models.detector import TADYOLODetector

__all__ = ["TADYOLODetector"]
__version__ = "0.1.0"


def __getattr__(name: str):
    """Keep package import lightweight so offline setup runs before Ultralytics."""
    if name == "TADYOLODetector":
        from tad_yolo.models.detector import TADYOLODetector

        return TADYOLODetector
    raise AttributeError(name)
