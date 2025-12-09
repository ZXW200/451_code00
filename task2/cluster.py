# K-Means clustering for image features
from pathlib import Path
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import matplotlib.pyplot as plt

from utils import save_js, fmt_time


# Find optimal number of clusters using elbow method
def find_k(feats, k_range=(2, 10), out_dir=None):
    print(f"Finding k in {k_range}")

    ks = range(k_range[0], k_range[1] + 1)
    loss = []
    scores = []

    # Test different k values and collect metrics
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(feats)

        loss.append(km.inertia_)
        scores.append(silhouette_score(feats, lbls))

        # Print progress for every 5th k value
        if k % 5 == 0:
            print(f"k={k}: loss={km.inertia_:.2f}, score={scores[-1]:.4f}")

    # Select k with best silhouette score
    best_k = ks[np.argmax(scores)]
    print(f"Best k: {best_k}")

    # Create elbow curve visualization if output directory provided
    if out_dir:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot inertia elbow curve
        ax1.plot(ks, loss, 'bo-')
        ax1.set_xlabel('k')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow')
        ax1.grid(True, alpha=0.3)

        # Plot silhouette scores with best k marked
        ax2.plot(ks, scores, 'ro-')
        ax2.axvline(x=best_k, color='g', linestyle='--', label=f'k={best_k}')
        ax2.set_xlabel('k')
        ax2.set_ylabel('Score')
        ax2.set_title('Silhouette')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'figures' / 'km_elbow.png', dpi=300, bbox_inches='tight')
        plt.close()

    return best_k


# Perform K-Means clustering with given k
def do_kmeans(feats, k):
    print(f"KMeans (k={k})")
    t0 = time.time()

    # Fit K-Means and get cluster labels
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbls = km.fit_predict(feats)

    dt = time.time() - t0
    print(f"Done: {fmt_time(dt)}")
    print(f"Sizes: {np.bincount(lbls)}")

    return lbls, km


# Evaluate clustering quality with multiple metrics
def eval_clus(feats, pred, true=None):
    n_c = len(np.unique(pred))

    # Subsample if dataset is too large for efficient metric calculation
    if len(feats) > 20000:
        print("Sampling for metrics...")
        idx = np.random.choice(len(feats), 20000, replace=False)
        fs = feats[idx]
        ps = pred[idx]
        ts = true[idx] if true is not None else None
    else:
        fs, ps, ts = feats, pred, true

    # Calculate clustering metrics
    m = {
        'n': n_c,
        'sil': silhouette_score(fs, ps),
        'db': davies_bouldin_score(fs, ps),
        'ch': calinski_harabasz_score(fs, ps)
    }

    # Add external validation metrics if ground truth provided
    if true is not None:
        m.update({
            'ari': adjusted_rand_score(ts, ps),
            'nmi': normalized_mutual_info_score(ts, ps)
        })

    print("Metrics:")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")

    return m


# Plot cluster size distribution
def plot_dist(lbls, name, path):
    unq, cnt = np.unique(lbls, return_counts=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(unq, cnt, alpha=0.7)

    # Add count labels on top of bars
    for bar, c in zip(bars, cnt):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{c}', ha='center', va='bottom')

    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title(f'{name} Dist')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


# Run complete clustering pipeline
def run_clus(feats, true, names, out_dir, fix_k=None):
    res = {}

    # Use fixed k or find optimal k
    if fix_k:
        print(f"Fixed k={fix_k}")
        k = fix_k
    else:
        k = find_k(
            feats,
            k_range=(2, min(10, len(names) + 3)),
            out_dir=out_dir
        )

    # Run K-Means clustering
    print(f"\nKMeans k={k}")
    lbls, mod = do_kmeans(feats, k)

    # Evaluate and store results
    met = eval_clus(feats, lbls, true)
    res['kmeans'] = {
        'labels': lbls.tolist(),
        'metrics': met,
        'k': k
    }

    # Plot cluster distribution
    plot_dist(
        lbls, 'KMeans',
        out_dir / 'figures' / 'clus_dist.png'
    )

    save_js(res, out_dir / 'results', 'clus_res.json')
    return res
