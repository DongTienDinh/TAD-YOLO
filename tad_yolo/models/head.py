"""Three-scale YOLOv8 detection head for D2, D3 and D4."""

from ultralytics.nn.modules import Detect


class ThreeScaleDetectionHead(Detect):
    """Legacy YOLOv8 Detect head operating at strides 4, 8 and 16."""

    legacy = True
