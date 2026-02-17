import torch
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.utils import resize
from .segformer_head import SegformerHead


@MODELS.register_module()
class SegFormerConvHead(SegformerHead):
    """Conv-based fusion replacing fusion_conv in SegformerHead."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        in_ch = self.fusion_conv.conv.in_channels
        out_ch = self.channels

        self.fuse_conv1 = ConvModule(
            in_ch, out_ch, kernel_size=1,
            norm_cfg=self.norm_cfg, act_cfg=dict(type='ReLU')
        )
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

        x = self.fuse_conv1(x)
        x = self.fuse_dwconv(x)
        x = self.fuse_conv2(x)

        x = self.cls_seg(x)
        return x
