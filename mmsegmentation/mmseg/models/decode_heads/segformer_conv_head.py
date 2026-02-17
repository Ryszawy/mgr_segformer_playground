import torch
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.utils import resize
from .segformer_head import SegformerHead


@MODELS.register_module()
class SegFormerConvHead(SegformerHead):
    """SegFormer head with conv-based fusion replacing fusion_conv."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # W SegFormerHead: fusion_conv przyjmuje concat z 4 skal:
        # in_channels = sum(self.in_channels) (czyli np. 32+64+160+256=512 dla B0)
        in_ch = sum(self.in_channels)
        out_ch = self.channels  # embed dim (np. 256)

        # Nasz blok fuse: 1x1 -> DW 3x3 -> 1x1
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
        # 1) dokładnie jak w SegformerHead.forward()
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
                    align_corners=self.align_corners))

        x = torch.cat(outs, dim=1)

        # 2) nasz fuse zamiast self.fusion_conv(...)
        x = self.fuse_conv1(x)
        x = self.fuse_dwconv(x)
        x = self.fuse_conv2(x)

        # 3) klasyfikator jak w bazie
        x = self.cls_seg(x)
        return x
