_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/default_runtime.py'
]

# modify num classes of the classification head
model = dict(
    head=dict(num_classes=120)
)

# checkpoint path
load_from = '../../checkpoints/swin_tiny_224_b16x64_300e.pth'

# only for test
train_cfg = None
val_cfg = None
test_cfg = dict()

# test dataset settings
test_dataloader = dict(
    dataset=dict(
        data_root='../../data/stanford_dogs',
        data_prefix=dict(img_path='test'),
        ann_file=None,
    )
)
