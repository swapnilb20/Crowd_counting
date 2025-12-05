import os, numpy as np
from PIL import Image
from tqdm import tqdm
from .adaptive_density import generate_density_map_adaptive
import torch

def create_densities(img_dir, gt_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    for img_name in tqdm(image_files):
        img_path = os.path.join(img_dir, img_name)
        txt_path = os.path.join(gt_dir, img_name.replace('.jpg', '.txt'))

        img = np.array(Image.open(img_path))
        pts, heads_wh = [], []

        with open(txt_path, 'r') as f:
            for line in f:
                x, y, w, h = map(float, line.strip().split(','))
                pts.append((int(x), int(y)))
                heads_wh.append((int(w), int(h)))

        density = generate_density_map_adaptive(
            img_shape=img.shape[:2],
            points=pts, heads_wh=heads_wh,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        np.savez_compressed(os.path.join(out_dir, img_name.replace('.jpg', '.npz')),
                            density=density.astype(np.float16))
