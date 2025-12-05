import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(model, dataset, n=6):
    imgs, true = next(iter(dataset))
    pred = model.predict(imgs)
    plt.figure(figsize=(12,5*n))
    for i in range(n):
        plt.subplot(n,3,3*i+1); plt.imshow(imgs[i]); plt.axis('off')
        plt.subplot(n,3,3*i+2); plt.imshow(true[i,...,0], cmap='jet'); plt.axis('off')
        plt.subplot(n,3,3*i+3); plt.imshow(pred[i,...,0], cmap='jet'); plt.axis('off')
    plt.tight_layout(); plt.show()
