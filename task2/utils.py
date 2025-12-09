import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch


def make_dir(name, base='history'):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = Path(base) / f'run_{name}_{ts}'
    path.mkdir(parents=True, exist_ok=True)

    (path / 'figures').mkdir(exist_ok=True)
    (path / 'features').mkdir(exist_ok=True)
    (path / 'models').mkdir(exist_ok=True)
    (path / 'results').mkdir(exist_ok=True)

    return path


def get_dev(dev=None):
    if dev is None:
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(dev)


def save_npy(data, path, name):
    p = path / name
    np.save(p, data)
    print(f"Saved: {p} ({data.shape})")


def load_npy(path):
    data = np.load(path)
    print(f"Loaded: {path} ({data.shape})")
    return data


def save_js(data, path, name):
    p = path / name

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


def load_js(path):
    with open(path, 'r') as f:
        return json.load(f)


def print_dev(dev):
    print(f"Device: {dev}")
    if dev.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")


def calc_bs(dev, cpu_bs=64, gpu_bs=256):
    if dev.type == 'cuda':
        return gpu_bs
    else:
        return cpu_bs


def fmt_time(sec):
    if sec < 60:
        return f"{sec:.2f}s"
    elif sec < 3600:
        return f"{sec / 60:.2f}m"
    else:
        return f"{sec / 3600:.2f}h"
