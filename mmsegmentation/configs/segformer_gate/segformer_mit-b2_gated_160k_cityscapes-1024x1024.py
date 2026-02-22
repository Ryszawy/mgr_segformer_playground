_base_ = '../segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py'

# =========================
#  SegFormer MiT-B2 + Gated Fusion Head
# =========================

model = dict(
    decode_head=dict(
        type='SegformerGatedHead',

        # MiT-B2 stage channels
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        input_transform='multiple_select',
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,  # Cityscapes

        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,

        # ---- gated fusion params ----
        gate_hidden=64,
        gate_share=False,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        ),
    )
)

# =========================
#  Train loop + scheduler (160k)
#  (override to keep everything consistent)
# =========================

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=4000)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
