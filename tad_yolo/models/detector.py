"""Complete fixed F4 detector with names aligned to the architecture diagram."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import torch
from torch import nn

from tad_yolo.models.backbone import ShallowHeavyBackbone
from tad_yolo.models.head import ThreeScaleDetectionHead
from tad_yolo.models.neck import DetectionFeatures, TerminalP2DySampleNeck


class TADYOLODetector(nn.Module):
    """F4 detector: shallow-heavy backbone + terminal-P2 DySample neck."""

    def __init__(
        self,
        num_classes: int = 10,
        stem_channels: int = 24,
        backbone_channels: Sequence[int] = (48, 64, 96, 128),
        neck_channels: Sequence[int] = (48, 64, 96),
        c2f_repeats: Sequence[int] = (1, 2, 2, 1),
        c2f_expansion: float = 0.5,
        sppf_kernel: int = 5,
        dysample: dict | None = None,
        bn_eps: float = 0.001,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model_name = "TAD-YOLO-F4"
        self.current_epoch = 0.0

        self.backbone = ShallowHeavyBackbone(
            stem_channels=stem_channels,
            stage_channels=backbone_channels,
            c2f_repeats=c2f_repeats,
            c2f_expansion=c2f_expansion,
            sppf_kernel=sppf_kernel,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )
        self.neck = TerminalP2DySampleNeck(
            backbone_channels=backbone_channels,
            neck_channels=neck_channels,
            dysample=dysample,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )
        self.detection_head = ThreeScaleDetectionHead(
            nc=num_classes,
            ch=tuple(neck_channels),
        )

        self.head_channels = tuple(neck_channels)
        self.strides = (4.0, 8.0, 16.0)
        self.stride = torch.tensor(self.strides)
        self.inplace = True
        self.max_det = 300
        self.task = "detect"
        self.yaml = {"nc": num_classes, "channels": 3, "model": self.model_name}
        self.names = {index: str(index) for index in range(num_classes)}
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self.criterion: Any = None

        self.detection_head.stride = self.stride
        self.detection_head.inplace = self.inplace
        if hasattr(self.detection_head, "max_det"):
            self.detection_head.max_det = self.max_det
        try:
            self.detection_head.bias_init()
        except Exception:
            pass

        # Ultralytics v8DetectionLoss locates Detect at model[-1].
        self.model = nn.ModuleList((self.backbone, self.neck, self.detection_head))

    @property
    def head(self) -> ThreeScaleDetectionHead:
        """Readable alias without registering the same module twice."""
        return self.detection_head

    def forward_features(self, images: torch.Tensor) -> DetectionFeatures:
        return self.neck(self.backbone(images))

    def predict(self, images: torch.Tensor) -> Any:
        return self.detection_head(list(self.forward_features(images)))

    def forward(self, x: torch.Tensor | dict[str, Any], *args, **kwargs) -> Any:
        return self.loss(x) if isinstance(x, dict) else self.predict(x)

    def init_criterion(self):
        from ultralytics.utils.loss import v8DetectionLoss

        return v8DetectionLoss(self)

    def loss(self, batch: dict[str, Any], predictions: Any = None):
        if self.criterion is None:
            self.criterion = self.init_criterion()
        if predictions is None:
            predictions = self.predict(batch["img"])
        return self.criterion(predictions, batch)

    def set_head_attr(self, **attributes: Any) -> None:
        """Forward trainer-controlled inference attributes to the Detect head."""
        for name, value in attributes.items():
            if hasattr(self.detection_head, name):
                setattr(self.detection_head, name, value)

    def _apply(self, fn):
        super()._apply(fn)
        self.stride = fn(self.stride)
        self.detection_head.stride = self.stride
        self.detection_head.anchors = fn(self.detection_head.anchors)
        self.detection_head.strides = fn(self.detection_head.strides)
        return self

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TADYOLODetector":
        model = config["model"]
        return cls(
            num_classes=int(model["num_classes"]),
            stem_channels=int(model["stem_channels"]),
            backbone_channels=tuple(model["backbone_channels"]),
            neck_channels=tuple(model["neck_channels"]),
            c2f_repeats=tuple(model["c2f_repeats"]),
            c2f_expansion=float(model["c2f_expansion"]),
            sppf_kernel=int(model["sppf_kernel"]),
            dysample=dict(model["dysample"]),
            bn_eps=float(model["bn_eps"]),
            bn_momentum=float(model["bn_momentum"]),
        )
