# Dimensionality reduction using UMAP algorithm
from pathlib import Path
import time
import numpy as np
import umap
import matplotlib.pyplot as plt

from utils import save_js, fmt_time


# Apply UMAP dimensionality reduction to features
def do_umap(feats, n_comp=50, n_neigh=15, dist=0.1, metric='euclidean'):
    print(f"UMAP: {feats.shape[1]} -> {n_comp}")

    t0 = time.time()

    # Configure UMAP reducer with specified parameters
    red = umap.UMAP(
        n_components=n_comp,
        n_neighbors=n_neigh,
        min_dist=dist,
        metric=metric,
        random_state=42,
        verbose=True,
        n_jobs=-1
    )

    # Fit and transform features to reduced dimensions
    res = red.fit_transform(feats)

    dt = time.time() - t0
    print(f"UMAP done: {fmt_time(dt)}")
    print(f"Shape: {res.shape}")

    return res, red


# Visualize 2D embedding with class labels
def plot_emb(emb, y, names, method, path):
    plt.figure(figsize=(12, 8))

    # Get unique labels and assign colors
    unq = np.unique(y)
    cols = plt.cm.tab10(np.linspace(0, 1, len(unq)))

    # Plot each class with different color
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


# Run dimensionality reduction pipeline
def run_dim_red(feats, lbls, names, out_dir):
    res = {}
    n = feats.shape[0]

    # Reduce to 50 dimensions for downstream tasks
    print("\nUMAP (50)")
    f_50, _ = do_umap(feats, n_comp=50, n_neigh=min(15, n - 1), dist=0.1)
    res['reduced'] = f_50

    # Reduce to 2 dimensions for visualization
    print("\nUMAP (2)")
    f_2, _ = do_umap(feats, n_comp=2, n_neigh=min(15, n - 1), dist=0.1)

    # Plot 2D visualization
    plot_emb(
        f_2, lbls, names, 'UMAP',
        out_dir / 'figures' / 'umap.png'
    )

    # Store reduction metadata
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
