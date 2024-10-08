_base_ = ['../oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py']

model = dict(
    bbox_head=dict(
        type='SharedCOBBRoIHead',
        ratio_type='none',
        score_type='cos',
        reg_dim=4,
        score_dim=4,
        ratio_dim=1,
        bbox_coder=dict(
            type='GVDeltaXYWHBBoxCoder',
            angle_norm_factor=None,
            target_means=(.0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2),
        ),
        cobb_coder=dict(type='COBBCoder', pow_iou=1.0, ratio_type='ln'),
        loss_ratio=dict(type="SmoothL1Loss", beta=0.05, loss_weight=0.2),
        loss_score=dict(type="SmoothL1Loss", beta=0.05, loss_weight=1.0)
    )
)

