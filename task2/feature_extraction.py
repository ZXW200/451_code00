from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import get_ext
from preprocessing import make_loader, plot_samples
from utils import save_js, fmt_time


def get_feats(model, dl, dev):
    print("Extracting...")

    model.eval()
    fs = []
    ls = []

    t0 = time.time()

    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(dl):
            imgs = imgs.to(dev)
            out = model(imgs)

            fs.append(out.cpu().numpy())
            ls.append(lbls.numpy())

            if (i + 1) % 10 == 0:
                print(f"Done {(i + 1) * dl.batch_size} imgs")

    feats = np.concatenate(fs, axis=0)
    lbls = np.concatenate(ls, axis=0)

    dt = time.time() - t0
    print(f"Done in {fmt_time(dt)}")
    print(f"Shape: {feats.shape}")

    return feats, lbls


def run_ext(data_path, ds_name, m_name, dev, bs, out_dir):
    print(f"Model: {m_name}")
    ext = get_ext(m_name, dev)
    size = ext.get_size()

    print("Data loader...")
    dl, info = make_loader(
        data_path, size, bs, shuf=False, aug=False, wk=4
    )

    p_viz = out_dir / 'figures' / 'samples.png'
    plot_samples(dl, min(16, len(dl.dataset)), p_viz, info['classes'])

    feats, lbls = get_feats(ext, dl, dev)

    stats = {
        'mean': np.mean(feats, axis=0).tolist(),
        'std': np.std(feats, axis=0).tolist(),
        'min': np.min(feats, axis=0).tolist(),
        'max': np.max(feats, axis=0).tolist()
    }

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
