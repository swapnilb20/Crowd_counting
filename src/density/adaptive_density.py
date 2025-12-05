import numpy as np
import torch
from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_kernel(sigma_rounded: float):
    sigma = float(sigma_rounded)
    size = int(6 * sigma) + 1
    half = size // 2

    xs = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(xs, xs)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    kernel /= (kernel.sum() + 1e-12)
    return torch.tensor(kernel, dtype=torch.float32)

def generate_density_map_adaptive(img_shape, points, heads_wh, device='cuda', cache=True, sigma_round=0.1):
    H, W = img_shape
    density = torch.zeros((H, W), dtype=torch.float32, device=device)

    if len(points) == 0:
        return density.cpu().numpy()

    pts = np.array(points, dtype=np.float32)
    head_wh = np.array(heads_wh, dtype=np.float32)
    sigmas = 0.3 * np.sqrt(head_wh[:, 0] * head_wh[:, 1])

    for (cx, cy), sigma in zip(pts, sigmas):
        if sigma <= 0: continue

        sigma_use = round(float(sigma), 1 if cache else 5)
        kernel = get_cached_kernel(sigma_use).to(device)

        ksize = kernel.shape[0]
        half = ksize // 2
        cx, cy = int(cx), int(cy)

        x1, x2 = max(0, cx - half), min(W, cx + half + 1)
        y1, y2 = max(0, cy - half), min(H, cy + half + 1)

        if x2 <= x1 or y2 <= y1: continue

        kx1, ky1 = half - (cx - x1), half - (cy - y1)
        kx2, ky2 = kx1 + (x2 - x1), ky1 + (y2 - y1)

        crop_kernel = kernel[ky1:ky2, kx1:kx2]
        density[y1:y2, x1:x2] += crop_kernel

    return density.cpu().numpy()
