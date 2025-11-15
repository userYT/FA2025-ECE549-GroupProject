_base_ = [
    '../_base_/models/swin_transformer/small_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]
load_from = 'checkpoints/swin_small.pth'

# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,          
        weight_decay=0.05,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=5.0)
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=50,         
        eta_min=1e-6
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,          
        save_best='accuracy/top1', 
        rule='greater'
    )
)

data_preprocessor = dict(
    num_classes=120,
    to_onehot=False,      
    batch_augments=None   
)

auto_scale_lr = None


