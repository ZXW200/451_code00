# Utility functions for deep learning pipeline
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch


# Create timestamped directory structure for pipeline outputs
def make_dir(name, base='history'):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = Path(base) / f'run_{name}_{ts}'
    path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different output types
    (path / 'figures').mkdir(exist_ok=True)
    (path / 'features').mkdir(exist_ok=True)
    (path / 'models').mkdir(exist_ok=True)
    (path / 'results').mkdir(exist_ok=True)

    return path


# Get available compute device, defaults to cuda if available
def get_dev(dev=None):
    if dev is None:
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(dev)


# Save numpy array to file
def save_npy(data, path, name):
    p = path / name
    np.save(p, data)
    print(f"Saved: {p} ({data.shape})")


# Load numpy array from file
def load_npy(path):
    data = np.load(path)
    print(f"Loaded: {path} ({data.shape})")
    return data


# Save data to JSON file with proper type conversion
def save_js(data, path, name):
    p = path / name

    # Convert numpy types to native Python types for JSON serialization
    def cvt(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: cvt(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cvt(i) for i in obj]
        else:
            return obj

    d = cvt(data)
    with open(p, 'w') as f:
        json.dump(d, f, indent=2)


# Load JSON file into dictionary
def load_js(path):
    with open(path, 'r') as f:
        return json.load(f)


# Print device information including GPU name if available
def print_dev(dev):
    print(f"Device: {dev}")
    if dev.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")


# Calculate appropriate batch size based on device type
def calc_bs(dev, cpu_bs=64, gpu_bs=256):
    if dev.type == 'cuda':
        return gpu_bs
    else:
        return cpu_bs


# Format seconds into human readable time string
def fmt_time(sec):
    if sec < 60:
        return f"{sec:.2f}s"
    elif sec < 3600:
        return f"{sec / 60:.2f}m"
    else:
        return f"{sec / 3600:.2f}h"
