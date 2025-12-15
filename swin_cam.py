import torch
import torch.nn.functional as F

def generate_swin_cam(model, img_tensor, target_class):
    """
    Grad-CAM for MMPretrain Swin Transformer.
    Correctly handles stage output as (x, hw_shape).
    """

    model.zero_grad()

    feats = []
    grads = []

    # Hook last stage
    target_layer = model.backbone.stages[-1]

    def forward_hook(module, inp, out):
        # out is (x, hw_shape)
        feats.append(out)

    def backward_hook(module, gin, gout):
        # gout is tuple aligned with out
        grads.append(gout)

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    logits = model(img_tensor)
    score = logits[:, target_class].sum()

    # Backward
    score.backward()

    h1.remove()
    h2.remove()

    # -------------------------------------------------
    # Parse outputs
    # -------------------------------------------------
    (x, hw_shape) = feats[0]        # x: (B, L, C)
    (gx, _) = grads[0]              # gx: gradient wrt x

    x = x.detach()[0]               # (L, C)
    gx = gx.detach()[0]             # (L, C)

    H, W = hw_shape                 # token grid

    # -------------------------------------------------
    # Grad-CAM on tokens
    # -------------------------------------------------
    weights = gx.mean(dim=0)        # (C,)
    cam = (x * weights).sum(dim=1)  # (L,)
    cam = F.relu(cam)

    cam = cam.reshape(H, W)

    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    return cam.cpu().numpy()
