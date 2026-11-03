"""DySample content-aware upsampling used at T4->T3 and T3->T2.

Adapted from the MIT-licensed reference implementation for:
"Learning to Upsample by Learning to Sample" (ICCV 2023).
See THIRD_PARTY_NOTICES.md for provenance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DySample(nn.Module):
    """Upsample by predicting grouped sampling offsets."""

    def __init__(
        self,
        in_channels: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        dyscope: bool = False,
        offset_scale: float = 0.25,
        offset_init_std: float = 0.001,
        sample_fp32: bool = True,
        sample_mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if style not in {"lp", "pl"}:
            raise ValueError(f"style must be 'lp' or 'pl', got {style!r}")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if style == "pl" and (in_channels < scale**2 or in_channels % scale**2):
            raise ValueError("style='pl' requires in_channels divisible by scale^2")

        self.scale = scale
        self.style = style
        self.groups = groups
        self.offset_scale = offset_scale
        self.sample_fp32 = sample_fp32
        self.sample_mode = sample_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

        conv_in = in_channels // scale**2 if style == "pl" else in_channels
        offset_channels = 2 * groups if style == "pl" else 2 * groups * scale**2
        self.offset = nn.Conv2d(conv_in, offset_channels, kernel_size=1, bias=True)
        nn.init.normal_(self.offset.weight, mean=0.0, std=offset_init_std)
        nn.init.zeros_(self.offset.bias)

        if dyscope:
            self.scope = nn.Conv2d(conv_in, offset_channels, kernel_size=1, bias=False)
            nn.init.zeros_(self.scope.weight)

        self.register_buffer("initial_positions", self._make_initial_positions())

    def _make_initial_positions(self) -> torch.Tensor:
        coordinate = torch.arange(
            (-self.scale + 1) / 2,
            (self.scale - 1) / 2 + 1,
            dtype=torch.float32,
        ) / self.scale
        grid = torch.stack(torch.meshgrid(coordinate, coordinate, indexing="ij")).transpose(1, 2)
        return grid.repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def _sample(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = offsets.shape
        offsets = offsets.reshape(batch, 2, -1, height, width)

        row = torch.arange(height, device=x.device, dtype=torch.float32) + 0.5
        col = torch.arange(width, device=x.device, dtype=torch.float32) + 0.5
        grid_row, grid_col = torch.meshgrid(row, col, indexing="ij")
        # [1, 2, 1, H, W], broadcast over grouped sub-pixel offsets.
        coordinates = torch.stack((grid_col, grid_row)).unsqueeze(0).unsqueeze(2)
        normalizer = torch.tensor(
            [width, height], device=x.device, dtype=torch.float32
        ).reshape(1, 2, 1, 1, 1)
        coordinates = 2.0 * (coordinates + offsets.float()) / normalizer - 1.0
        coordinates = F.pixel_shuffle(
            coordinates.reshape(batch, -1, height, width), self.scale
        )
        coordinates = (
            coordinates.reshape(batch, 2, -1, self.scale * height, self.scale * width)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )

        sample_input = x.float() if self.sample_fp32 else x
        sample_grid = coordinates.float() if self.sample_fp32 else coordinates.to(x.dtype)
        output = F.grid_sample(
            sample_input.reshape(batch * self.groups, -1, height, width),
            sample_grid,
            mode=self.sample_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return output.reshape(batch, -1, self.scale * height, self.scale * width).to(x.dtype)

    def _predict_lp_offsets(self, x: torch.Tensor) -> torch.Tensor:
        layer_input = x if x.dtype == self.offset.weight.dtype else x.to(self.offset.weight.dtype)
        raw_offsets = self.offset(layer_input)
        if hasattr(self, "scope"):
            return raw_offsets * self.scope(layer_input).sigmoid() * 0.5 + self.initial_positions
        return raw_offsets * self.offset_scale + self.initial_positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.style == "pl":
            layer_input = x if x.dtype == self.offset.weight.dtype else x.to(self.offset.weight.dtype)
            shuffled = F.pixel_shuffle(layer_input, self.scale)
            raw_offsets = self.offset(shuffled)
            if hasattr(self, "scope"):
                raw_offsets = raw_offsets * self.scope(shuffled).sigmoid() * 0.5
            else:
                raw_offsets = raw_offsets * self.offset_scale
            offsets = F.pixel_unshuffle(raw_offsets, self.scale) + self.initial_positions
            return self._sample(x, offsets)
        offsets = self._predict_lp_offsets(x)
        return self._sample(x, offsets)
