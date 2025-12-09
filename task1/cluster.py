from pathlib import Path
import hdbscan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from utils import save_fig


class KM_Algo:
    def __init__(self, k_range=(2, 10), seed=42):
        self.k_range = k_range
        self.seed = seed
        self.model = None
        self.best_k = None
        self.loss = {}
        self.scores = {}

    def find_k(self, X, out_path):
        print("Finding k...")

        for k in range(self.k_range[0], self.k_range[1] + 1):
            km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
            lbls = km.fit_predict(X)
            self.loss[k] = km.inertia_
            self.scores[k] = silhouette_score(X, lbls)

            print(f"k={k}: loss={km.inertia_:.2f}, score={self.scores[k]:.3f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ks = list(self.loss.keys())
        ax1.plot(ks, list(self.loss.values()), 'bo-')
        ax1.set_xlabel('k')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow')
        ax1.grid(True, alpha=0.3)

        ax2.plot(ks, list(self.scores.values()), 'ro-')
        ax2.set_xlabel('k')
        ax2.set_ylabel('Score')
        ax2.set_title('Silhouette')
        ax2.grid(True, alpha=0.3)

        self.best_k = max(self.scores, key=self.scores.get)
        ax2.axvline(x=self.best_k, color='g', linestyle='--',
                    label=f'Best k={self.best_k}')
        ax2.legend()

        plt.tight_layout()
        save_fig(fig, out_path, '05_elbow')

        print(f"Best k: {self.best_k}")

        return self.best_k

    def fit(self, X, k=None):
        if k is None:
            k = self.best_k

        print(f"Fitting KMeans (k={k})...")

        self.model = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        self.model.fit(X)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def eval(self, X, y):
        res = {
            'sil': silhouette_score(X, y),
            'db': davies_bouldin_score(X, y),
            'ch': calinski_harabasz_score(X, y),
            'loss': self.model.inertia_
        }

        print("KM Stats:")
        print(f"  Sil: {res['sil']:.4f}")
        print(f"  DB: {res['db']:.4f}")
        print(f"  CH: {res['ch']:.2f}")
        print(f"  Loss: {res['loss']:.2f}")

        return res


class HDB_Algo:
    def __init__(self, min_size=5, min_samp=None, metric='euclidean'):
        self.min_size = min_size
        self.min_samp = min_samp if min_samp else min_size
        self.metric = metric
        self.model = None

    def fit(self, X):
        print(f"Fitting HDB (sz={self.min_size}, smp={self.min_samp})...")

        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_size,
            min_samples=self.min_samp,
            metric=self.metric
        )
        self.model.fit(X)

        return self

    def predict(self, X):
        return self.model.labels_

    def eval(self, X, y):
        mask = y != -1
        X_c = X[mask]
        y_c = y[mask]

        n_c = len(set(y)) - (1 if -1 in y else 0)
        n_noise = list(y).count(-1)

        res = {
            'n_clusters': n_c,
            'n_noise': n_noise,
            'noise_pct': (n_noise / len(y) * 100)
        }

        if n_c >= 2 and len(y_c) > 0:
            res['sil'] = silhouette_score(X_c, y_c)
            res['db'] = davies_bouldin_score(X_c, y_c)
            res['ch'] = calinski_harabasz_score(X_c, y_c)
        else:
            res['sil'] = 0.0
            res['db'] = 0.0
            res['ch'] = 0.0

        print("HDB Stats:")
        print(f"  Clusters: {n_c}")
        print(f"  Noise: {n_noise} ({res['noise_pct']:.2f}%)")
        if n_c >= 2:
            print(f"  Sil: {res['sil']:.4f}")
            print(f"  DB: {res['db']:.4f}")
            print(f"  CH: {res['ch']:.2f}")

        return res


def plot_clusters(X, y, name, out_path):
    print(f"Plotting {name}...")

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    pca = PCA(n_components=2, random_state=42)
    xp = pca.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    xt = tsne.fit_transform(X)

    sc1 = ax[0].scatter(xp[:, 0], xp[:, 1], c=y, cmap='viridis',
                               alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    ax[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    ax[0].set_title(f'{name} PCA')
    ax[0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=ax[0])

    sc2 = ax[1].scatter(xt[:, 0], xt[:, 1], c=y, cmap='viridis',
                               alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax[1].set_xlabel('t-SNE 1')
    ax[1].set_ylabel('t-SNE 2')
    ax[1].set_title(f'{name} t-SNE')
    ax[1].grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=ax[1])

    plt.tight_layout()
    save_fig(fig, out_path, f'06_{name.lower()}_viz')


def compare_res(res, out_path):
    print("Comparing...")

    methods = list(res.keys())
    sils = [res[m].get('sil', 0) for m in methods]
    dbs = [res[m].get('db', 0) for m in methods]
    chs = [res[m].get('ch', 0) for m in methods]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].bar(methods, sils, color=['#1f77b4', '#ff7f0e'])
    ax[0].set_ylabel('Score')
    ax[0].set_title('Silhouette (High good)')
    ax[0].grid(True, alpha=0.3, axis='y')

    ax[1].bar(methods, dbs, color=['#1f77b4', '#ff7f0e'])
    ax[1].set_ylabel('Score')
    ax[1].set_title('DB (Low good)')
    ax[1].grid(True, alpha=0.3, axis='y')

    ax[2].bar(methods, chs, color=['#1f77b4', '#ff7f0e'])
    ax[2].set_ylabel('Score')
    ax[2].set_title('CH (High good)')
    ax[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_fig(fig, out_path, '07_compare')


def analyze_clusters(df, y, name):
    print(f"Analyzing {name}")

    res = {}
    c_ids = sorted(set(y))

    for cid in c_ids:
        if cid == -1:
            cnt = sum(y == -1)
            pct = cnt / len(y) * 100
            print(f"Noise: {cnt} ({pct:.2f}%)")
            res['noise'] = {'count': cnt, 'pct': pct}
            continue

        mask = y == cid
        sub = df[mask]

        stats = {
            'count': len(sub),
            'pct': len(sub) / len(df) * 100,
            'mean': sub.mean().to_dict(),
            'std': sub.std().to_dict()
        }

        res[f'c_{cid}'] = stats

        print(f"C {cid}: {len(sub)} ({stats['pct']:.2f}%)")

    return res


def run_cluster(prep_path, out_path):
    import pandas as pd
    from utils import save_json, save_csv

    print("Loading data")
    p_dir = Path(prep_path)
    f_path = p_dir / 'clean_data.csv'

    if not f_path.exists():
        print(f"Error: {f_path}")
        raise FileNotFoundError("File missing")

    df = pd.read_csv(f_path)
    print(f"Shape: {df.shape}")

    print("\nK-Means")

    km = KM_Algo(k_range=(2, 10), seed=42)
    best_k = km.find_k(df, out_path)
    km.fit(df, k=best_k)
    y_km = km.predict(df)
    m_km = km.eval(df, y_km)
    plot_clusters(df, y_km, 'KMeans', out_path)

    print("\nHDBSCAN")

    hdb = HDB_Algo(min_size=15, min_samp=5)
    hdb.fit(df)
    y_hdb = hdb.predict(df)
    m_hdb = hdb.eval(df, y_hdb)
    plot_clusters(df, y_hdb, 'HDBSCAN', out_path)

    print("\nAnalysis")

    all_res = {
        'KMeans': m_km,
        'HDBSCAN': m_hdb
    }

    compare_res(all_res, out_path)
    a_km = analyze_clusters(df, y_km, 'KMeans')
    a_hdb = analyze_clusters(df, y_hdb, 'HDBSCAN')

    final = {
        'kmeans': {
            'k': best_k,
            'metrics': m_km,
            'labels': y_km.tolist(),
            'analysis': a_km
        },
        'hdbscan': {
            'metrics': m_hdb,
            'labels': y_hdb.tolist(),
            'analysis': a_hdb
        }
    }

    save_json(final, out_path, 'cluster_res.json')

    df_out = df.copy()
    df_out['km'] = y_km
    df_out['hdb'] = y_hdb
    save_csv(df_out, out_path, 'clustered_data.csv')

    c_stats = {
        'km': a_km,
        'hdb': a_hdb
    }
    save_json(c_stats, out_path, 'cluster_stats.json')

    print(f"\nDone. Output: {out_path}")
