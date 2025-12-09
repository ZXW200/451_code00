# Feature extraction from images using pretrained deep learning models
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import get_ext
from preprocessing import make_loader, plot_samples
from utils import save_js, fmt_time


# Extract features from all images in data loader using model
def get_feats(model, dl, dev):
    print("Extracting...")

    model.eval()
    fs = []
    ls = []

    t0 = time.time()

    # Process all batches without computing gradients
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(dl):
            imgs = imgs.to(dev)
            out = model(imgs)

            # Store features and labels
            fs.append(out.cpu().numpy())
            ls.append(lbls.numpy())

            # Print progress every 10 batches
            if (i + 1) % 10 == 0:
                print(f"Done {(i + 1) * dl.batch_size} imgs")

    # Concatenate all batches
    feats = np.concatenate(fs, axis=0)
    lbls = np.concatenate(ls, axis=0)

    dt = time.time() - t0
    print(f"Done in {fmt_time(dt)}")
    print(f"Shape: {feats.shape}")

    return feats, lbls


# Main feature extraction pipeline
def run_ext(data_path, ds_name, m_name, dev, bs, out_dir):
    print(f"Model: {m_name}")
    # Load feature extractor model
    ext = get_ext(m_name, dev)
    size = ext.get_size()

    # Create data loader for the dataset
    print("Data loader...")
    dl, info = make_loader(
        data_path, size, bs, shuf=False, aug=False, wk=4
    )

    # Visualize sample images from dataset
    p_viz = out_dir / 'figures' / 'samples.png'
    plot_samples(dl, min(16, len(dl.dataset)), p_viz, info['classes'])

    # Extract features from all images
    feats, lbls = get_feats(ext, dl, dev)

    # Calculate feature statistics
    stats = {
        'mean': np.mean(feats, axis=0).tolist(),
        'std': np.std(feats, axis=0).tolist(),
        'min': np.min(feats, axis=0).tolist(),
        'max': np.max(feats, axis=0).tolist()
    }

    # Store extraction metadata
    meta = {
        'ds': ds_name,
        'path': str(data_path),
        'model': m_name,
        'dim': ext.get_dim(),
        'size': size,
        'bs': bs,
        'n': len(feats),
        'n_cls': info['n_cls'],
        'classes': info['classes'],
        'shape': feats.shape,
        'stats': stats,
        'dev': str(dev)
    }

    save_js(meta, out_dir / 'results', 'ext_meta.json')

    return feats, lbls, meta
