import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.utils import resize
from .segformer_head import SegformerHead


@MODELS.register_module()
class SegFormerConvHead(SegformerHead):
    """SegFormer head with conv-based fusion replacing fusion_conv.
    Uses LazyConv2d to infer concat channel dim at runtime (version-robust).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        out_ch = self.channels  # embed dim used by cls_seg

        # 1x1 (lazy in_channels) -> DW 3x3 -> 1x1
        self.fuse_conv1 = nn.LazyConv2d(out_ch, kernel_size=1, bias=False)

        # norm/act around lazy conv (keep simple & robust)
        # Use the same norm type as in the base head via ConvModule for later blocks.
        # For the first block we add BN+ReLU explicitly.
        # NOTE: If you use SyncBN on 1 GPU, it still works but BN is fine too.
        self.fuse_bn1 = nn.BatchNorm2d(out_ch)
        self.fuse_act1 = nn.ReLU(inplace=True)

        self.fuse_dwconv = ConvModule(
            out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch,
            norm_cfg=self.norm_cfg, act_cfg=dict(type='ReLU')
        )
        self.fuse_conv2 = ConvModule(
            out_ch, out_ch, kernel_size=1,
            norm_cfg=self.norm_cfg, act_cfg=dict(type='ReLU')
        )

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            )

        x = torch.cat(outs, dim=1)

        # Lazy 1x1 + BN + ReLU
        x = self.fuse_conv1(x)
        x = self.fuse_bn1(x)
        x = self.fuse_act1(x)

        x = self.fuse_dwconv(x)
        x = self.fuse_conv2(x)

        x = self.cls_seg(x)
        return x
