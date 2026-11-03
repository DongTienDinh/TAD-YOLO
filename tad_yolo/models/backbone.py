"""Shallow-heavy semantic backbone corresponding to B2-B5 in the diagram."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import nn

from tad_yolo.nn.blocks import C2f, ConvBNAct, SPPF


class BackboneFeatures(NamedTuple):
    """Backbone outputs at strides 4, 8, 16 and 32."""

    b2: torch.Tensor
    b3: torch.Tensor
    b4: torch.Tensor
    b5: torch.Tensor


class ShallowHeavyBackbone(nn.Module):
    """YOLOv8n-style backbone with 24/48/64/96/128 channel allocation."""

    def __init__(
        self,
        stem_channels: int = 24,
        stage_channels: Sequence[int] = (48, 64, 96, 128),
        c2f_repeats: Sequence[int] = (1, 2, 2, 1),
        c2f_expansion: float = 0.5,
        sppf_kernel: int = 5,
        bn_eps: float = 0.001,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 4 or len(c2f_repeats) != 4:
            raise ValueError("stage_channels and c2f_repeats must each contain four values")
        b2_channels, b3_channels, b4_channels, b5_channels = stage_channels
        r2, r3, r4, r5 = c2f_repeats
        common = {"bn_eps": bn_eps, "bn_momentum": bn_momentum}

        self.output_channels = tuple(stage_channels)
        self.stem = ConvBNAct(3, stem_channels, 3, 2, 1, **common)
        self.stage2 = nn.Sequential(
            ConvBNAct(stem_channels, b2_channels, 3, 2, 1, **common),
            C2f(b2_channels, b2_channels, r2, shortcut=True, expansion=c2f_expansion, **common),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(b2_channels, b3_channels, 3, 2, 1, **common),
            C2f(b3_channels, b3_channels, r3, shortcut=True, expansion=c2f_expansion, **common),
        )
        self.stage4 = nn.Sequential(
            ConvBNAct(b3_channels, b4_channels, 3, 2, 1, **common),
            C2f(b4_channels, b4_channels, r4, shortcut=True, expansion=c2f_expansion, **common),
        )
        self.stage5 = nn.Sequential(
            ConvBNAct(b4_channels, b5_channels, 3, 2, 1, **common),
            C2f(b5_channels, b5_channels, r5, shortcut=True, expansion=c2f_expansion, **common),
            SPPF(b5_channels, b5_channels, sppf_kernel, **common),
        )

    def forward(self, images: torch.Tensor) -> BackboneFeatures:
        stem = self.stem(images)
        b2 = self.stage2(stem)
        b3 = self.stage3(b2)
        b4 = self.stage4(b3)
        b5 = self.stage5(b4)
        return BackboneFeatures(b2=b2, b3=b3, b4=b4, b5=b5)
