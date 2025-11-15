_base_ = [
    '../_base_/models/mobilevit/mobilevit_s.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/imagenet_bs256.py',
]

model = dict(
    head=dict(
        num_classes=120
    )
)

# no normalize for original implements
data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[0, 0, 0],
    std=[1, 1, 1],
    # use bgr directly
    to_rgb=False,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=288, edge='short'),
    dict(type='CenterCrop', crop_size=256),
    dict(type='PackInputs'),
]

dataset_type = 'CustomDataset'
data_root = 'data/stanford_dogs'

dataset_type = 'CustomDataset'
data_root = 'data/stanford_dogs'

train_cfg = dict(
    by_epoch=True,
    max_epochs=100,
    val_interval=1
)

train_dataloader = dict(
    batch_size=32,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        data_root=f'{data_root}/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=256),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackInputs'),
        ],
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        data_root=f'{data_root}/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=288, edge='short'),
            dict(type='CenterCrop', crop_size=256),
            dict(type='PackInputs'),
        ],
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = val_dataloader

val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_evaluator = dict(type='Accuracy', topk=(1, 5))



optim_wrapper = dict(
    optimizer=dict(
        lr=5e-3,       
        weight_decay=0.0001,
        type='SGD',
        momentum=0.9,
    ),
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,          
        save_best='accuracy/top1', 
        rule='greater'
    )
)
