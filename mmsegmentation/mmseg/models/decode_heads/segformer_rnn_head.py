from typing import List

import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize


@MODELS.register_module()
class SegformerRNNHead(BaseDecodeHead):
    """SegFormer-style head with GRU/LSTM refinement on a downsampled spatial sequence.

    Pipeline:
      - per-stage 1x1 conv -> channels
      - resize to same spatial size (usually 1/4)
      - concat -> fuse 1x1 to channels
      - downsample spatially (seq_downsample)
      - flatten H'*W' to sequence, run GRU/LSTM
      - project back to channels, reshape
      - upsample back to 1/4
      - classifier
    """

    def __init__(
        self,
        interpolate_mode: str = 'bilinear',
        rnn_type: str = 'GRU',
        hidden_size: int = 128,
        num_layers: int = 1,
        seq_downsample: int = 4,
        **kwargs,
    ):
        # IMPORTANT: multi-level inputs
        kwargs.setdefault('input_transform', 'multiple_select')
        super().__init__(**kwargs)

        assert isinstance(self.in_channels, (list, tuple)), \
            "SegformerRNNHead expects in_channels as list/tuple (multi-scale)."

        self.interpolate_mode = interpolate_mode
        self.seq_downsample = int(seq_downsample)
        assert self.seq_downsample >= 1, "seq_downsample must be >= 1"
        assert rnn_type in ['GRU', 'LSTM'], "rnn_type must be 'GRU' or 'LSTM'"

        # 1) project each stage to self.channels
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

        # 2) fuse after concat (like SegFormer)
        self.fuse = ConvModule(
            in_channels=self.channels * len(self.in_channels),
            out_channels=self.channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        # 3) RNN over sequence (batch_first=True)
        rnn_dropout = 0.0 if num_layers <= 1 else 0.1
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
                bidirectional=False,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=self.channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
                bidirectional=False,
            )

        # 4) project hidden -> channels
        self.rnn_proj = nn.Linear(hidden_size, self.channels)

        # 5) downsample module (avg pool)
        if self.seq_downsample > 1:
            self.pool = nn.AvgPool2d(kernel_size=self.seq_downsample, stride=self.seq_downsample)
        else:
            self.pool = nn.Identity()

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == len(self.in_channels), \
            f"Expected {len(self.in_channels)} inputs, got {len(inputs)}."

        target_size = inputs[0].shape[2:]  # usually 1/4 resolution
        feats = []

        # project + resize each scale to target
        for i, x in enumerate(inputs):
            x = self.convs[i](x)
            if x.shape[2:] != target_size:
                x = resize(
                    input=x,
                    size=target_size,
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners,
                )
            feats.append(x)

        # concat + fuse -> [B, C, H, W]
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)

        # downsample -> [B, C, H', W']
        x_small = self.pool(x)

        B, C, Hs, Ws = x_small.shape
        L = Hs * Ws

        # flatten -> sequence [B, L, C]
        seq = x_small.flatten(2).transpose(1, 2).contiguous()

        # RNN -> [B, L, hidden]
        out, _ = self.rnn(seq)

        # proj -> [B, L, C]
        out = self.rnn_proj(out)

        # reshape -> [B, C, H', W']
        out = out.transpose(1, 2).contiguous().view(B, C, Hs, Ws)

        # upsample back to target [B, C, H, W]
        if out.shape[2:] != target_size:
            out = resize(
                input=out,
                size=target_size,
                mode=self.interpolate_mode,
                align_corners=self.align_corners,
            )

        # classifier -> [B, num_classes, H, W]
        out = self.cls_seg(out)
        return out
