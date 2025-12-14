import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import mmcv
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    # top_k_accuracy_score,
    confusion_matrix,
    classification_report
)

from mmpretrain.apis import ImageClassificationInferencer
from mmpretrain import init_model
from captum.attr import IntegratedGradients

# =========================
# 1. Configurations
# =========================

CONFIG = 'configs/swin_transformer/swin-tiny_test.py'
CHECKPOINT = 'checkpoints/swin_tiny_224.pth'
TEST_ROOT = 'data/stanford_dogs/test'

OUT_DIR = 'visualization/outputs'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# =========================
# 2. Load model & inferencer
# =========================

print('Loading model...')
model = init_model(CONFIG, CHECKPOINT, device=DEVICE)
model.eval()

inferencer = ImageClassificationInferencer(
    model=model,
    device=DEVICE,
    progress=False
)

# =========================
# 3. Collect test predictions
# =========================

print('Running inference on test set...')

y_true, y_pred, y_score = [], [], []
img_paths = []

class_names = sorted(os.listdir(TEST_ROOT))
all_images = []

for cls_idx, cls_name in enumerate(class_names):
    cls_dir = os.path.join(TEST_ROOT, cls_name)
    if not os.path.isdir(cls_dir):
        continue
    for img_name in os.listdir(cls_dir):
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            all_images.append((cls_idx, os.path.join(cls_dir, img_name)))

'''
# Optional: Subsample test set for faster visualization
import random

TEST_RATIO = 0.025  
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
random.shuffle(all_images)

num_test = int(len(all_images) * TEST_RATIO)
all_images = all_images[:num_test]
print(f'Using {num_test} / {len(class_names)} test images '
      f'({TEST_RATIO * 100:.0f}% of test set)')
'''

for cls_idx, img_path in tqdm(all_images, desc='Inference on test set'):
    result = inferencer(img_path)[0]

    y_true.append(cls_idx)
    y_pred.append(result['pred_label'])

    with torch.no_grad():
        data = list(inferencer.preprocess([img_path]))[0]
        logits = model.test_step(data)[0].pred_score.cpu().numpy()

    y_score.append(logits)
    img_paths.append(img_path)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.stack(y_score)

# =========================
# Fixed samples for visualization
# =========================

correct_indices = [
    i for i in range(len(y_true))
    if y_true[i] == y_pred[i]
]

VIS_INDICES = correct_indices[:3]
print('Visualization sample indices:', VIS_INDICES)

# =========================
# 4. Metrics
# =========================

top1 = accuracy_score(y_true, y_pred)
top5 = np.mean([
    y_true[i] in np.argsort(y_score[i])[-5:]
    for i in range(len(y_true))
])

report = classification_report(
    y_true,
    y_pred,
    digits=4
)

with open(os.path.join(OUT_DIR, 'metrics.txt'), 'w') as f:
    f.write(f'Top-1 Accuracy: {top1:.4f}\n')
    f.write(f'Top-5 Accuracy: {top5:.4f}\n\n')
    f.write(report)

print('Metrics saved.')

# =========================
# 5. Confusion Matrix
# =========================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
plt.title('Confusion Matrix (Stanford Dogs)')
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'))
plt.close()

print('Confusion matrix saved.')

# =========================
# 6. Custom Grad-CAM for Swin (with data_preprocessor)
# =========================

print('Running custom Grad-CAM for Swin (data_preprocessor aligned)...')

from swin_cam import generate_swin_cam 

cam_dir = os.path.join(OUT_DIR, 'cam_attention')
os.makedirs(cam_dir, exist_ok=True)

for vis_id, i in enumerate(VIS_INDICES):
    img_bgr = mmcv.imread(img_paths[i])            # (H, W, 3), BGR
    img_rgb = mmcv.bgr2rgb(img_bgr)                
    H, W = img_rgb.shape[:2]

    img_resized = mmcv.imresize(img_rgb, (224, 224))

    img_tensor = (
        torch.tensor(img_resized)
        .permute(2, 0, 1)
        .float()
        .to(DEVICE)
    )

    with torch.no_grad():
        data = {
            'inputs': [img_tensor],
            'data_samples': None
        }
        data = model.data_preprocessor(data, training=False)
        img_tensor_norm = data['inputs']

    target_class = int(y_pred[i])

    cam = generate_swin_cam(
        model=model,
        img_tensor=img_tensor_norm,
        target_class=target_class
    )

    cam = mmcv.imresize(cam, (W, H))
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    heatmap = plt.get_cmap('jet')(cam)[:, :, :3]
    overlay = 0.5 * (img_rgb / 255.0) + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.imsave(
        os.path.join(cam_dir, f'swin_cam_{vis_id}.png'),
        overlay
    )

print('Custom Swin Grad-CAM (data_preprocessor aligned) saved.')

# =========================
# 7. Integrated Gradients
# =========================

print('Running Integrated Gradients...')

ig_root = os.path.join(OUT_DIR, 'integrated_gradients')
ig_raw_dir = os.path.join(ig_root, 'raw')
ig_overlay_dir = os.path.join(ig_root, 'overlay')
ig_image_dir = os.path.join(ig_root, 'image')

os.makedirs(ig_raw_dir, exist_ok=True)
os.makedirs(ig_overlay_dir, exist_ok=True)
os.makedirs(ig_image_dir, exist_ok=True)

ig = IntegratedGradients(model)

for vis_id, i in enumerate(VIS_INDICES):
    img = mmcv.imread(img_paths[i])
    img_float = img.astype(np.float32) / 255.0
    H, W = img.shape[:2]

    img_resized = mmcv.imresize(img, (224, 224))

    img_tensor = (
        torch.tensor(img_resized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    baseline = torch.zeros_like(img_tensor)

    attr = ig.attribute(
        img_tensor,
        baseline,
        target=int(y_pred[i])
    )

    attr = attr.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    attr = np.abs(attr).mean(axis=2)

    attr -= attr.min()
    attr /= (attr.max() + 1e-8)

    attr_full = mmcv.imresize(attr, (W, H))
    heatmap = plt.get_cmap('jet')(attr_full)[:, :, :3]
    overlay = 0.6 * img_float + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.imsave(os.path.join(ig_raw_dir, f'ig_raw_{vis_id}.png'), attr_full, cmap='hot')
    plt.imsave(os.path.join(ig_overlay_dir, f'ig_overlay_{vis_id}.png'), overlay)
    plt.imsave(os.path.join(ig_image_dir, f'ig_image_{vis_id}.png'), img_float)

print('Integrated Gradients visualizations saved.')

print('✅ All visualizations completed successfully!')
