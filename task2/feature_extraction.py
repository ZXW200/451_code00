from pathlib import Path
from typing import Tuple, Dict, Any
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import create_feature_extractor
from preprocessing import create_single_dataloader, visualize_samples
from utils import save_numpy, save_json, format_time


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    print("Extracting features...")
    
    model.eval()
    features_list = []
    labels_list = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            features = model(images)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * dataloader.batch_size} images")
    
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    
    elapsed_time = time.time() - start_time
    print(f"Features extracted in {format_time(elapsed_time)}")
    print(f"Shape: {features_array.shape}")
    print(f"Avg time/image: {elapsed_time / len(labels_array):.4f}s")
    
    return features_array, labels_array


def run_feature_extraction(
    dataset_path: str,
    dataset_name: str,
    model_name: str,
    device: torch.device,
    batch_size: int,
    output_dir: Path
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    print(f"Loading model: {model_name}")
    extractor = create_feature_extractor(model_name, device)
    input_size = extractor.get_input_size()
    
    print("Loading dataset...")
    dataloader, dataset_info = create_single_dataloader(
        dataset_path=dataset_path,
        input_size=input_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        num_workers=4
    )
    
    sample_output_path = output_dir / 'figures' / 'sample_images_grid.png'
    visualize_samples(
        dataloader=dataloader,
        num_samples=min(16, len(dataloader.dataset)),
        output_path=sample_output_path,
        class_names=dataset_info['class_names']
    )
    
    features, labels = extract_features(
        model=extractor,
        dataloader=dataloader,
        device=device
    )
    
    feature_stats = {
        'mean': np.mean(features, axis=0).tolist(),
        'std': np.std(features, axis=0).tolist(),
        'min': np.min(features, axis=0).tolist(),
        'max': np.max(features, axis=0).tolist()
    }
    
    metadata = {
        'dataset_name': dataset_name,
        'dataset_path': str(dataset_path),
        'model_name': model_name,
        'feature_dim': extractor.get_feature_dim(),
        'input_size': input_size,
        'batch_size': batch_size,
        'num_samples': len(features),
        'num_classes': dataset_info['num_classes'],
        'class_names': dataset_info['class_names'],
        'features_shape': features.shape,
        'labels_shape': labels.shape,
        'feature_stats': feature_stats,
        'device': str(device)
    }
    
    save_json(metadata, output_dir / 'results', 'extraction_metadata.json')
    
    return features, labels, metadata


def load_extracted_features(
    features_path: Path,
    labels_path: Path,
    metadata_path: Path
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    print("Loading features...")
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    if metadata_path and metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            'num_classes': len(np.unique(labels)),
            'num_samples': len(features),
            'feature_dim': features.shape[1] if len(features.shape) > 1 else 1
        }
    
    print(f"Features: {features.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Classes: {metadata.get('num_classes', 'Unknown')}")
    print(f"Model: {metadata.get('model_name', 'Unknown')}")
    
    return features, labels, metadata
