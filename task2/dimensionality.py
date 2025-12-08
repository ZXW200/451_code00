from pathlib import Path
from typing import Tuple, List, Dict, Any
import time
import numpy as np
import umap
import matplotlib.pyplot as plt

from utils import save_numpy, save_json, format_time


def apply_umap(
    features: np.ndarray,
    n_components: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    verbose: bool = True
) -> Tuple[np.ndarray, umap.UMAP]:
    print(f"UMAP: {features.shape[1]} -> {n_components} dimensions")
    
    start_time = time.time()
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        verbose=verbose,
        n_jobs=-1
    )
    
    features_reduced = reducer.fit_transform(features)
    
    elapsed_time = time.time() - start_time
    print(f"UMAP completed in {format_time(elapsed_time)}")
    print(f"Shape: {features_reduced.shape}")
    
    return features_reduced, reducer


def visualize_2d_embedding(
    features_2d: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    method_name: str,
    output_path: Path
):
    plt.figure(figsize=(12, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=class_name,
            alpha=0.7,
            s=20
        )
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'{method_name} Visualization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_dimensionality_reduction(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_dir: Path
) -> Dict[str, Any]:
    results = {}
    n_samples = features.shape[0]
    
    print("\nUMAP (50 components for clustering)")
    features_reduced, reducer = apply_umap(
        features, 
        n_components=50,
        n_neighbors=min(15, n_samples - 1),
        min_dist=0.1
    )
    results['reduced_features'] = features_reduced
    
    print("\nUMAP (2 components for visualization)")
    features_2d, reducer_2d = apply_umap(
        features,
        n_components=2,
        n_neighbors=min(15, n_samples - 1),
        min_dist=0.1
    )
    
    visualize_2d_embedding(
        features_2d,
        labels,
        class_names,
        'UMAP',
        output_dir / 'figures' / 'umap_visualization.png'
    )
    
    reduction_metadata = {
        'method': 'UMAP',
        'original_features_shape': list(features.shape),
        'reduced_50_shape': list(features_reduced.shape),
        'reduced_2d_shape': list(features_2d.shape),
        'n_samples': int(n_samples)
    }
    
    save_json(reduction_metadata, output_dir / 'results', 'dimensionality_metrics.json')
    
    results['metadata'] = reduction_metadata
    return results
