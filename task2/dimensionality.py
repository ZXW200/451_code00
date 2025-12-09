from pathlib import Path
import time
import numpy as np
import umap
import matplotlib.pyplot as plt

from utils import save_js, fmt_time


def do_umap(feats, n_comp=50, n_neigh=15, dist=0.1, metric='euclidean'):
    print(f"UMAP: {feats.shape[1]} -> {n_comp}")

    t0 = time.time()

    red = umap.UMAP(
        n_components=n_comp,
        n_neighbors=n_neigh,
        min_dist=dist,
        metric=metric,
        random_state=42,
        verbose=True,
        n_jobs=-1
    )

    res = red.fit_transform(feats)

    dt = time.time() - t0
    print(f"UMAP done: {fmt_time(dt)}")
    print(f"Shape: {res.shape}")

    return res, red


def plot_emb(emb, y, names, method, path):
    plt.figure(figsize=(12, 8))

    unq = np.unique(y)
    cols = plt.cm.tab10(np.linspace(0, 1, len(unq)))

    for i, lb in enumerate(unq):
        mask = y == lb
        cn = names[lb] if lb < len(names) else f'C_{lb}'
        plt.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[cols[i]], label=cn, alpha=0.7, s=20
        )

    plt.xlabel('C 1')
    plt.ylabel('C 2')
    plt.title(f'{method} Viz')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def run_dim_red(feats, lbls, names, out_dir):
    res = {}
    n = feats.shape[0]

    print("\nUMAP (50)")
    f_50, _ = do_umap(feats, n_comp=50, n_neigh=min(15, n - 1), dist=0.1)
    res['reduced'] = f_50

    print("\nUMAP (2)")
    f_2, _ = do_umap(feats, n_comp=2, n_neigh=min(15, n - 1), dist=0.1)

    plot_emb(
        f_2, lbls, names, 'UMAP',
        out_dir / 'figures' / 'umap.png'
    )

    meta = {
        'method': 'UMAP',
        'shape_orig': list(feats.shape),
        'shape_50': list(f_50.shape),
        'shape_2': list(f_2.shape),
        'n': int(n)
    }

    save_js(meta, out_dir / 'results', 'dim_meta.json')

    res['meta'] = meta
    return res
