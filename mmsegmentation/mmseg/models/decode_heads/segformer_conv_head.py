import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .segformer_head import SegformerHead


@MODELS.register_module()
class SegFormerConvHead(SegformerHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        embed_dim = self.channels

        # zastępujemy linearny fuse MLP
        self.fuse_conv1 = ConvModule(
            embed_dim,
            embed_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU')
        )

        self.fuse_dwconv = ConvModule(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU')
        )

        self.fuse_conv2 = ConvModule(
            embed_dim,
            embed_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU')
        )

    def forward(self, inputs):
        # to jest oryginalny flow SegformerHead
        x = self._forward_feature(inputs)

        # NASZ NOWY FUSE BLOK
        x = self.fuse_conv1(x)
        x = self.fuse_dwconv(x)
        x = self.fuse_conv2(x)

        x = self.cls_seg(x)

        return x
