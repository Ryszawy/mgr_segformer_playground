_base_ = '../segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py'

model = dict(
    decode_head=dict(
        type='SegformerRNNHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        rnn_type='GRU',
        hidden_size=128,
        num_layers=1,
        seq_downsample=4,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    )
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=4000)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=80000, by_epoch=False),
]
