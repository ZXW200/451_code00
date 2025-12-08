from pathlib import Path
from typing import Dict, Optional, Tuple
import hdbscan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from utils import save_figure


class KMeansClusterer:
    def __init__(self, k_range: Tuple[int, int] = (2, 10), random_state: int = 42):
        self.k_range = k_range
        self.random_state = random_state
        self.model = None
        self.optimal_k = None
        self.inertias = {}
        self.silhouette_scores = {}
        
    def find_optimal_k(self, X: pd.DataFrame, output_dir: Path) -> int:
        print("Finding optimal k...")
        
        for k in range(self.k_range[0], self.k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            self.inertias[k] = kmeans.inertia_
            self.silhouette_scores[k] = silhouette_score(X, labels)
            
            print(f"k={k}: inertia={kmeans.inertia_:.2f}, silhouette={self.silhouette_scores[k]:.3f}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        k_values = list(self.inertias.keys())
        ax1.plot(k_values, list(self.inertias.values()), 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(k_values, list(self.silhouette_scores.values()), 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs k')
        ax2.grid(True, alpha=0.3)
        
        self.optimal_k = max(self.silhouette_scores, key=self.silhouette_scores.get)
        ax2.axvline(x=self.optimal_k, color='g', linestyle='--', 
                   label=f'Optimal k={self.optimal_k}')
        ax2.legend()
        
        plt.tight_layout()
        save_figure(fig, output_dir, '05_kmeans_elbow')
        
        print(f"Optimal k: {self.optimal_k}")
        
        return self.optimal_k
    
    def fit(self, X: pd.DataFrame, k: Optional[int] = None) -> 'KMeansClusterer':
        if k is None:
            k = self.optimal_k
        
        print(f"Fitting K-Means (k={k})...")
        
        self.model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        self.model.fit(X)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        metrics = {
            'silhouette_score': silhouette_score(X, labels),
            'davies_bouldin_score': davies_bouldin_score(X, labels),
            'calinski_harabasz_score': calinski_harabasz_score(X, labels),
            'inertia': self.model.inertia_
        }
        
        print("K-Means metrics:")
        print(f"  Silhouette: {metrics['silhouette_score']:.4f}")
        print(f"  Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
        print(f"  Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}")
        print(f"  Inertia: {metrics['inertia']:.2f}")
        
        return metrics


class HDBSCANClusterer:
    def __init__(self, min_cluster_size: int = 5, min_samples: Optional[int] = None,
                 metric: str = 'euclidean'):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples else min_cluster_size
        self.metric = metric
        self.model = None
        
    def fit(self, X: pd.DataFrame) -> 'HDBSCANClusterer':
        print(f"Fitting HDBSCAN (size={self.min_cluster_size}, samples={self.min_samples})...")
        
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric
        )
        self.model.fit(X)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.labels_
    
    def evaluate(self, X: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_percentage': (n_noise / len(labels) * 100)
        }
        
        if n_clusters >= 2 and len(labels_filtered) > 0:
            metrics['silhouette_score'] = silhouette_score(X_filtered, labels_filtered)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_filtered, labels_filtered)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_filtered, labels_filtered)
        else:
            metrics['silhouette_score'] = 0.0
            metrics['davies_bouldin_score'] = 0.0
            metrics['calinski_harabasz_score'] = 0.0
        
        print("HDBSCAN metrics:")
        print(f"  Clusters: {n_clusters}")
        print(f"  Noise: {n_noise} ({metrics['noise_percentage']:.2f}%)")
        if n_clusters >= 2:
            print(f"  Silhouette: {metrics['silhouette_score']:.4f}")
            print(f"  Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
            print(f"  Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}")
        
        return metrics


def visualize_clusters(X: pd.DataFrame, labels: np.ndarray, method_name: str,
                      output_dir: Path) -> None:
    print(f"Visualizing {method_name}...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
                              alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    axes[0].set_title(f'{method_name} Clusters (PCA)')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis',
                              alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title(f'{method_name} Clusters (t-SNE)')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    
    plt.tight_layout()
    filename = f'06_{method_name.lower().replace(" ", "_")}_clusters_visualization'
    save_figure(fig, output_dir, filename)


def compare_clustering_methods(results: Dict[str, Dict], output_dir: Path) -> None:
    print("Comparing methods...")
    
    methods = list(results.keys())
    silhouette_scores = [results[m].get('silhouette_score', 0) for m in methods]
    davies_bouldin_scores = [results[m].get('davies_bouldin_score', 0) for m in methods]
    calinski_scores = [results[m].get('calinski_harabasz_score', 0) for m in methods]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].bar(methods, silhouette_scores, color=['#1f77b4', '#ff7f0e'])
    axes[0].set_ylabel('Score')
    axes[0].set_title('Silhouette Score (Higher is Better)')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(methods, davies_bouldin_scores, color=['#1f77b4', '#ff7f0e'])
    axes[1].set_ylabel('Score')
    axes[1].set_title('Davies-Bouldin Score (Lower is Better)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(methods, calinski_scores, color=['#1f77b4', '#ff7f0e'])
    axes[2].set_ylabel('Score')
    axes[2].set_title('Calinski-Harabasz Score (Higher is Better)')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, output_dir, '07_clustering_comparison')


def analyze_cluster_characteristics(df: pd.DataFrame, labels: np.ndarray, 
                                  method_name: str) -> dict:
    print(f"{method_name} analysis")
    
    analysis = {}
    unique_clusters = sorted(set(labels))
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            noise_count = sum(labels == -1)
            noise_percentage = noise_count / len(labels) * 100
            print(f"Noise: {noise_count} ({noise_percentage:.2f}%)")
            analysis['noise'] = {'size': noise_count, 'percentage': noise_percentage}
            continue
        
        mask = labels == cluster_id
        cluster_data = df[mask]
        
        cluster_stats = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100,
            'mean_values': cluster_data.mean().to_dict(),
            'std_values': cluster_data.std().to_dict()
        }
        
        analysis[f'cluster_{cluster_id}'] = cluster_stats
        
        print(f"Cluster {cluster_id}: {len(cluster_data)} ({cluster_stats['percentage']:.2f}%)")
    
    return analysis


def run_clustering_task(preprocess_output: str, output_dir: Path) -> None:
    import pandas as pd
    from utils import save_results, save_dataframe
    
    print("[1/4] Loading data")
    preprocess_dir = Path(preprocess_output)
    preprocessed_file = preprocess_dir / 'data_preprocessed.csv'
    
    if not preprocessed_file.exists():
        print(f"Error: file not found: {preprocessed_file}")
        raise FileNotFoundError(f"Required file not found: {preprocessed_file}")
    
    df_normalized = pd.read_csv(preprocessed_file)
    print(f"Shape: {df_normalized.shape}")
    
    print("\n[2/4] K-Means")
    
    kmeans = KMeansClusterer(k_range=(2, 10), random_state=42)
    optimal_k = kmeans.find_optimal_k(df_normalized, output_dir)
    kmeans.fit(df_normalized, k=optimal_k)
    kmeans_labels = kmeans.predict(df_normalized)
    kmeans_metrics = kmeans.evaluate(df_normalized, kmeans_labels)
    visualize_clusters(df_normalized, kmeans_labels, 'K-Means', output_dir)
    
    print("\n[3/4] HDBSCAN")
    
    hdbscan = HDBSCANClusterer(min_cluster_size=15, min_samples=5)
    hdbscan.fit(df_normalized)
    hdbscan_labels = hdbscan.predict(df_normalized)
    hdbscan_metrics = hdbscan.evaluate(df_normalized, hdbscan_labels)
    visualize_clusters(df_normalized, hdbscan_labels, 'HDBSCAN', output_dir)
    
    print("\n[4/4] Analysis")
    
    clustering_results = {
        'K-Means': kmeans_metrics,
        'HDBSCAN': hdbscan_metrics
    }
    
    compare_clustering_methods(clustering_results, output_dir)
    kmeans_analysis = analyze_cluster_characteristics(df_normalized, kmeans_labels, 'K-Means')
    hdbscan_analysis = analyze_cluster_characteristics(df_normalized, hdbscan_labels, 'HDBSCAN')
    
    results = {
        'kmeans': {
            'optimal_k': optimal_k,
            'metrics': kmeans_metrics,
            'labels': kmeans_labels.tolist(),
            'analysis': kmeans_analysis
        },
        'hdbscan': {
            'metrics': hdbscan_metrics,
            'labels': hdbscan_labels.tolist(),
            'analysis': hdbscan_analysis
        }
    }
    
    save_results(results, output_dir, 'clustering_results.json')
    
    df_with_clusters = df_normalized.copy()
    df_with_clusters['kmeans_cluster'] = kmeans_labels
    df_with_clusters['hdbscan_cluster'] = hdbscan_labels
    save_dataframe(df_with_clusters, output_dir, 'data_with_clusters.csv')
    
    cluster_analysis_results = {
        'kmeans_clusters': kmeans_analysis,
        'hdbscan_clusters': hdbscan_analysis
    }
    save_results(cluster_analysis_results, output_dir, 'cluster_analysis.json')
    
    print(f"\nComplete. Output: {output_dir}")

