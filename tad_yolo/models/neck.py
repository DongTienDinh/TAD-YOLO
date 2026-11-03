"""Terminal-P2 neck with DySample top-down fusion and one bottom-up path."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from tad_yolo.models.backbone import BackboneFeatures
from tad_yolo.nn.blocks import C2f, ConvBNAct
from tad_yolo.nn.upsampling import DySample


class DetectionFeatures(NamedTuple):
    """Features consumed by detection branches D2, D3 and D4."""

    d2: torch.Tensor
    d3: torch.Tensor
    d4: torch.Tensor


class TerminalP2DySampleNeck(nn.Module):
    """F4 neck: terminal T2, DySample at 4->3 and 3->2, no N5/D5."""

    def __init__(
        self,
        backbone_channels: Sequence[int] = (48, 64, 96, 128),
        neck_channels: Sequence[int] = (48, 64, 96),
        dysample: dict | None = None,
        bn_eps: float = 0.001,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()
        if len(backbone_channels) != 4 or len(neck_channels) != 3:
            raise ValueError("Expected B2-B5 and T2-T4 channel definitions")
        b2_channels, b3_channels, b4_channels, b5_channels = backbone_channels
        t2_channels, t3_channels, t4_channels = neck_channels
        common = {"bn_eps": bn_eps, "bn_momentum": bn_momentum}
        dy = dict(dysample or {})

        self.output_channels = tuple(neck_channels)

        # B5 --Nearest x2--> concat(B4) --> T4
        self.t4_fusion = C2f(
            b5_channels + b4_channels, t4_channels,
            repeats=1, shortcut=False, expansion=0.5, **common,
        )

        # T4 --DySample x2--> concat(B3) --> T3
        self.t4_to_t3_upsample = DySample(t4_channels, **dy)
        self.t3_fusion = C2f(
            t4_channels + b3_channels, t3_channels,
            repeats=1, shortcut=False, expansion=0.5, **common,
        )

        # T3 --DySample x2--> concat(B2) --> terminal T2
        self.t3_to_t2_upsample = DySample(t3_channels, **dy)
        self.t2_fusion = C2f(
            t3_channels + b2_channels, t2_channels,
            repeats=1, shortcut=False, expansion=0.5, **common,
        )

        # Only retained bottom-up route: T3 --stride-2--> concat(T4) --> N4
        self.t3_to_n4_downsample = ConvBNAct(
            t3_channels, t4_channels, 3, 2, 1, **common,
        )
        self.n4_fusion = C2f(
            2 * t4_channels, t4_channels,
            repeats=1, shortcut=False, expansion=0.5, **common,
        )

    @staticmethod
    def _concat(upsampled: torch.Tensor, skip: torch.Tensor, route: str) -> torch.Tensor:
        if upsampled.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                f"Spatial mismatch on {route}: {tuple(upsampled.shape[-2:])} "
                f"vs {tuple(skip.shape[-2:])}. Input height/width must be divisible by 32."
            )
        return torch.cat((upsampled, skip), dim=1)

    def forward(self, features: BackboneFeatures) -> DetectionFeatures:
        b2, b3, b4, b5 = features

        upsampled_b5 = F.interpolate(b5, size=b4.shape[-2:], mode="nearest")
        t4 = self.t4_fusion(self._concat(upsampled_b5, b4, "B5->T4"))

        upsampled_t4 = self.t4_to_t3_upsample(t4)
        t3 = self.t3_fusion(self._concat(upsampled_t4, b3, "T4->T3"))

        upsampled_t3 = self.t3_to_t2_upsample(t3)
        t2 = self.t2_fusion(self._concat(upsampled_t3, b2, "T3->T2"))

        downsampled_t3 = self.t3_to_n4_downsample(t3)
        n4 = self.n4_fusion(self._concat(downsampled_t3, t4, "T3->N4"))

        # D2=T2 and D3=T3 are terminal/direct; D4=N4. There is no N5/D5.
        return DetectionFeatures(d2=t2, d3=t3, d4=n4)
