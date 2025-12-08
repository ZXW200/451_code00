from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
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

from utils import save_numpy, save_json, format_time


def find_optimal_k_kmeans(
        features: np.ndarray,
        k_range: Tuple[int, int] = (2, 10),
        output_dir: Optional[Path] = None
) -> int:
    print(f"Finding optimal k in range {k_range}")

    k_values = range(k_range[0], k_range[1] + 1)
    inertias = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features, labels))

        if k % 5 == 0:
            print(f"k={k}: inertia={kmeans.inertia_:.2f}, silhouette={silhouette_scores[-1]:.4f}")

    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"Optimal k: {optimal_k}")

    if output_dir:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(k_values, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)

        ax2.plot(k_values, silhouette_scores, 'ro-')
        ax2.axvline(x=optimal_k, color='g', linestyle='--',
                    label=f'Optimal k={optimal_k}')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'figures' / 'kmeans_elbow_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    return optimal_k


def perform_kmeans_clustering(
        features: np.ndarray,
        n_clusters: int
) -> Tuple[np.ndarray, KMeans]:
    print(f"K-Means clustering (k={n_clusters})")

    start_time = time.time()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    elapsed_time = time.time() - start_time
    print(f"Completed in {format_time(elapsed_time)}")
    print(f"Cluster sizes: {np.bincount(labels)}")

    return labels, kmeans


def evaluate_clustering(
        features: np.ndarray,
        cluster_labels: np.ndarray,
        labels_true: Optional[np.ndarray] = None
) -> Dict[str, float]:
    n_clusters = len(np.unique(cluster_labels))

    # 针对大数据集的优化：采样计算指标
    if len(features) > 20000:
        print("Warning: Dataset too large, computing metrics on sample...")
        indices = np.random.choice(len(features), 20000, replace=False)
        features_sample = features[indices]
        labels_sample = cluster_labels[indices]
        labels_true_sample = labels_true[indices] if labels_true is not None else None
    else:
        features_sample = features
        labels_sample = cluster_labels
        labels_true_sample = labels_true

    metrics = {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_score(features_sample, labels_sample),
        'davies_bouldin_score': davies_bouldin_score(features_sample, labels_sample),
        'calinski_harabasz_score': calinski_harabasz_score(features_sample, labels_sample)
    }

    if labels_true is not None:
        metrics.update({
            'adjusted_rand_score': adjusted_rand_score(labels_true_sample, labels_sample),
            'normalized_mutual_info': normalized_mutual_info_score(labels_true_sample, labels_sample)
        })

    print("Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    return metrics


def plot_cluster_distribution(
        cluster_labels: np.ndarray,
        method_name: str,
        output_path: Path
):
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique_labels, counts, alpha=0.7)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{count}', ha='center', va='bottom')

    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.title(f'{method_name} - Cluster Distribution')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_clustering_pipeline(
        features: np.ndarray,
        labels_true: Optional[np.ndarray],
        class_names: List[str],
        output_dir: Path,
        fixed_k: Optional[int] = None  # 新增参数：如果传入则直接使用，不搜索
) -> Dict[str, Any]:
    results = {}

    if fixed_k is not None:
        print(f"1. Using pre-determined optimal k={fixed_k} (Skipping search)")
        optimal_k = fixed_k
    else:
        print("1. Finding optimal k")
        optimal_k = find_optimal_k_kmeans(
            features,
            k_range=(2, min(10, len(class_names) + 3)),
            output_dir=output_dir
        )

    print(f"\n2. K-Means clustering (k={optimal_k})")
    kmeans_labels, kmeans_model = perform_kmeans_clustering(features, optimal_k)

    kmeans_metrics = evaluate_clustering(features, kmeans_labels, labels_true)
    results['kmeans'] = {
        'labels': kmeans_labels.tolist(),
        'metrics': kmeans_metrics,
        'optimal_k': optimal_k
    }

    plot_cluster_distribution(
        kmeans_labels,
        'K-Means',
        output_dir / 'figures' / 'cluster_distribution.png'
    )

    save_json(results, output_dir / 'results', 'clustering_metrics.json')

    return results
