_base_ = '../segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py'

model = dict(
    decode_head=dict(
        type='SegFormerConvHead'
    )
)
