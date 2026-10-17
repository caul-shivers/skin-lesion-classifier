import numpy as np
import torch
from PIL import Image


def gradcam(model, x, target_class=None):
    """x: (1, 3, H, W) preprocessed tensor.
    Returns (probs: list[float], cam: np.ndarray in [0, 1] of shape (h, w), target_class)."""
    store = {}
    handle = model.backbone.inception5b.register_forward_hook(
        lambda m, i, o: store.setdefault("acts", o)
    )
    try:
        # All backbone parameters are frozen, so autograd would build no graph at all.
        # Making the input require grad forces the graph to exist.
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            if target_class is None:
                target_class = int(probs.argmax())
            acts = store["acts"]
            grads = torch.autograd.grad(logits[0, target_class], acts)[0]
    finally:
        handle.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1))[0].detach()
    cam = cam.numpy()
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)
    return probs.detach().tolist(), cam, target_class


def _jet(v):
    """Minimal 'jet' colormap: v in [0,1] -> uint8 RGB (blue -> green -> red)."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend the heatmap onto the original (un-normalized) image."""
    img = img.convert("RGB")
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC)
    heat = _jet(np.asarray(cam_img, dtype=np.float32) / 255.0)
    # Fade out cold (low-importance) regions so the lesion stays visible.
    a = (np.asarray(cam_img, dtype=np.float32) / 255.0)[..., None] * alpha * 2
    a = np.clip(a, 0, alpha + 0.2)
    blended = np.asarray(img, dtype=np.float32) * (1 - a) + heat.astype(np.float32) * a
    return Image.fromarray(blended.astype(np.uint8))
