import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch


def create_output_dir(task_name: str, base_dir: str = 'history') -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / f'instance_{task_name}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'features').mkdir(exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)
    
    return output_dir


def get_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device_obj = torch.device(device)
    
    return device_obj


def save_numpy(data: np.ndarray, output_dir: Path, filename: str) -> None:
    filepath = output_dir / filename
    np.save(filepath, data)
    print(f"Saved: {filepath} ({data.shape})")


def load_numpy(filepath: Path) -> np.ndarray:
    data = np.load(filepath)
    print(f"Loaded: {filepath} ({data.shape})")
    return data


def save_json(data: Dict[str, Any], output_dir: Path, filename: str) -> None:
    filepath = output_dir / filename
    
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    data = convert_types(data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Path) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


# def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
#     dataset_configs = {
#         'cats_dogs': {
#             'name': 'Cats vs Dogs',
#             'num_classes': 2,
#             'class_names': ['cat', 'dog'],
#             'url': 'https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset',
#             'local_path': 'datasets/cats_dogs'
#         },
#         'oxford_pets': {
#             'name': 'Oxford-IIIT Pet Dataset',
#             'num_classes': 37,
#             'class_names': None,
#             'url': 'https://www.robots.ox.ac.uk/~vgg/data/pets/',
#             'local_path': 'datasets/oxford_pets'
#         },
#         'food101': {
#             'name': 'Food-101',
#             'num_classes': 101,
#             'class_names': None,
#             'url': 'https://www.kaggle.com/datasets/dansbecker/food-101',
#             'local_path': 'datasets/food101'
#         }
#     }
    
#     if dataset_name not in dataset_configs:
#         raise ValueError(f"Unknown dataset: {dataset_name}")
    
#     return dataset_configs[dataset_name]


def print_device_info(device: torch.device) -> None:
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")


def calculate_batch_size(device: torch.device, default_cpu: int = 64, default_gpu: int = 256) -> int:
    if device.type == 'cuda':
        return default_gpu
    else:
        return default_cpu


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


# def create_class_mapping(class_names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
#     name_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}
#     idx_to_name = {idx: name for name, idx in name_to_idx.items()}
    
#     return name_to_idx, idx_to_name
