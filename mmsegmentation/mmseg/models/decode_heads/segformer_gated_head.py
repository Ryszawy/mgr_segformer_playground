# Copyright (c) OpenMMLab.
# SegFormer Gated Fusion decode head
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize


@MODELS.register_module()
class SegformerGatedHead(BaseDecodeHead):
    """
    SegFormer-like decode head with per-scale gated fusion.

    Pipeline:
      - per-stage 1x1 conv to 'channels'
      - upsample to same spatial size (usually 1/4)
      - gate each scale: GAP -> MLP -> sigmoid -> multiply
      - concat -> fuse -> classifier
    """

    def __init__(
        self,
        interpolate_mode: str = 'bilinear',
        gate_hidden: int = 64,
        gate_share: bool = False,
        **kwargs,
    ):
        """
        Args:
            interpolate_mode: resize mode for upsampling features.
            gate_hidden: hidden size in the gate MLP.
            gate_share: if True, share a single MLP across all scales (lighter).
            **kwargs: standard BaseDecodeHead kwargs, notably:
                - in_channels (List[int]) length=4 for SegFormer
                - channels (int) unified embed dim
                - num_classes, dropout_ratio, norm_cfg, act_cfg, etc.
        """
        super().__init__(**kwargs)

        assert isinstance(self.in_channels, (list, tuple)), \
            "SegformerGatedHead expects in_channels as a list/tuple (multi-scale)."
        self.interpolate_mode = interpolate_mode

        # 1) per-scale projection to self.channels
        self.convs = nn.ModuleList()
        for ch in self.in_channels:
            self.convs.append(
                ConvModule(
                    in_channels=ch,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )

        # 2) gate MLP(s): GAP -> (Linear -> ReLU -> Linear) -> sigmoid
        def make_gate_mlp():
            return nn.Sequential(
                nn.Linear(self.channels, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, 1),
            )

        self.gate_share = gate_share
        if gate_share:
            self.gate_mlp = make_gate_mlp()
        else:
            self.gate_mlps = nn.ModuleList([make_gate_mlp() for _ in range(len(self.in_channels))])

        self.gap = nn.AdaptiveAvgPool2d(1)

        # 3) fuse after concat
        self.fuse = ConvModule(
            in_channels=self.channels * len(self.in_channels),
            out_channels=self.channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def _apply_gate(self, feat: torch.Tensor, idx: int) -> torch.Tensor:
        """
        feat: [B, C, H, W]
        returns gated feat: [B, C, H, W]
        """
        # Global pooling -> [B, C]
        pooled = self.gap(feat).flatten(1)

        # MLP -> [B, 1]
        if self.gate_share:
            gate_logits = self.gate_mlp(pooled)
        else:
            gate_logits = self.gate_mlps[idx](pooled)

        gate = torch.sigmoid(gate_logits).view(-1, 1, 1, 1)  # [B,1,1,1]
        return feat * gate

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        inputs: list of multi-level features (e.g. 4 stages from MiT)
        """
        assert len(inputs) == len(self.in_channels), \
            f"Expected {len(self.in_channels)} inputs, got {len(inputs)}."

        # Project + resize all to the spatial size of the highest-resolution feature (usually stage1)
        projected = []
        target_size = inputs[0].shape[2:]  # (H, W) of the first stage (1/4 for SegFormer)

        for i, x in enumerate(inputs):
            x = self.convs[i](x)  # -> [B, channels, Hi, Wi]
            if x.shape[2:] != target_size:
                x = resize(
                    input=x,
                    size=target_size,
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners,
                )
            # Apply gate BEFORE concatenation
            x = self._apply_gate(x, i)
            projected.append(x)

        x = torch.cat(projected, dim=1)   # [B, channels*4, H, W]
        x = self.fuse(x)                  # [B, channels, H, W]
        x = self.cls_seg(x)               # [B, num_classes, H, W]
        return x
