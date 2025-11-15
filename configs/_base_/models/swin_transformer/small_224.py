# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='small',
        img_size=224,
        drop_path_rate=0.1,   
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=120,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss'),  
        cal_acc=True,                    
    ),

    train_cfg=dict(augments=[]), 
)

