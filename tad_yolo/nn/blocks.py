"""Small, explicit YOLOv8 building blocks used by the F4 graph."""

from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        activation: bool = True,
        bn_eps: float = 0.001,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """YOLO bottleneck used inside C2f."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
        bn_eps: float = 0.001,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(
            in_channels, hidden_channels, 3, 1, 1,
            bn_eps=bn_eps, bn_momentum=bn_momentum,
        )
        self.conv2 = ConvBNAct(
            hidden_channels, out_channels, 3, 1, 1, groups=groups,
            bn_eps=bn_eps, bn_momentum=bn_momentum,
        )
        self.use_shortcut = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.use_shortcut else y


class C2f(nn.Module):
    """YOLOv8 CSP bottleneck with two outer convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int = 1,
        shortcut: bool = False,
        groups: int = 1,
        expansion: float = 0.5,
        bn_eps: float = 0.001,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        self.split = ConvBNAct(
            in_channels, 2 * self.hidden_channels, 1, 1, 0,
            bn_eps=bn_eps, bn_momentum=bn_momentum,
        )
        self.blocks = nn.ModuleList(
            Bottleneck(
                self.hidden_channels,
                self.hidden_channels,
                shortcut=shortcut,
                groups=groups,
                expansion=1.0,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
            )
            for _ in range(repeats)
        )
        self.fuse = ConvBNAct(
            (2 + repeats) * self.hidden_channels, out_channels, 1, 1, 0,
            bn_eps=bn_eps, bn_momentum=bn_momentum,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = list(self.split(x).chunk(2, dim=1))
        parts.extend(block(parts[-1]) for block in self.blocks)
        return self.fuse(torch.cat(parts, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast used at backbone stage B5."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        bn_eps: float = 0.001,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.reduce = ConvBNAct(
            in_channels, hidden_channels, 1, 1, 0,
            bn_eps=bn_eps, bn_momentum=bn_momentum,
        )
        self.fuse = ConvBNAct(
            4 * hidden_channels, out_channels, 1, 1, 0,
            bn_eps=bn_eps, bn_momentum=bn_momentum,
        )
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.reduce(x)
        x1 = self.pool(x0)
        x2 = self.pool(x1)
        x3 = self.pool(x2)
        return self.fuse(torch.cat((x0, x1, x2, x3), dim=1))
