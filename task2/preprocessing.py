from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np


# Custom image dataset class that loads images from directory structure
class ImgDS(Dataset):
    def __init__(self, root, transform=None, exts=('.jpg', '.jpeg', '.png', '.bmp')):
        self.root = Path(root)
        self.tf = transform
        self.exts = exts
        self.data = []
        self.classes = []
        self.c2i = {}
        self._load()

    # Load all image paths and labels from directory structure
    def _load(self):
        # Each subdirectory is a class
        c_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if not c_dirs: raise ValueError(f"No dirs in {self.root}")

        # Map class names to indices
        self.classes = [d.name for d in c_dirs]
        self.c2i = {n: i for i, n in enumerate(self.classes)}

        # Collect all image paths with their labels
        for c_dir in c_dirs:
            idx = self.c2i[c_dir.name]
            for ext in self.exts:
                for p in c_dir.glob(f'*{ext}'):
                    self.data.append((str(p), idx))

        if not self.data: raise ValueError(f"No imgs in {self.root}")

    # Return number of samples in dataset
    def __len__(self):
        return len(self.data)

    # Load and return single image with label
    def __getitem__(self, idx):
        p, lbl = self.data[idx]
        try:
            # Load image and convert to RGB
            img = Image.open(p).convert('RGB')
            if self.tf: img = self.tf(img)
            return img, lbl
        except Exception as e:
            # Return zero tensor if image loading fails
            print(f"Err: {p} - {e}")
            return torch.zeros((3, 224, 224)), lbl

    # Get list of class names
    def get_classes(self):
        return self.classes

    # Get number of classes
    def get_n_classes(self):
        return len(self.classes)


# Create image transforms with optional data augmentation
def get_trans(size=224, aug=False):
    # ImageNet normalization values
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if aug:
        # Training transforms with augmentation
        t = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(), norm
        ])
    else:
        # Validation and test transforms without augmentation
        t = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(), norm
        ])
    return t


# Create train, validation and test data loaders
def make_loaders(path, size, bs, tr=0.7, vr=0.15, te=0.15, wk=4):
    # Verify split ratios sum to 1
    if abs(tr + vr + te - 1.0) > 1e-6: raise ValueError("Split sum != 1")
    print(f"Load: {path}")

    # Create dataset with transforms
    tf = get_trans(size, aug=False)
    ds = ImgDS(path, transform=tf)
    print(f"N: {len(ds)}, C: {ds.get_n_classes()}")

    # Calculate split sizes
    n_tot = len(ds)
    n_tr = int(tr * n_tot)
    n_val = int(vr * n_tot)
    n_te = n_tot - n_tr - n_val
    print(f"Split: {n_tr}/{n_val}/{n_te}")

    # Split dataset into train, validation and test
    ds_tr, ds_val, ds_te = random_split(ds, [n_tr, n_val, n_te], generator=torch.Generator().manual_seed(42))
    ds_tr.dataset.transform = get_trans(size, aug=True)

    # Create data loaders for each split
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=wk, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=wk, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=wk, pin_memory=True)

    # Return loaders and dataset info
    info = {
        'n_cls': ds.get_n_classes(), 'classes': ds.get_classes(),
        'n_tot': n_tot, 'n_tr': n_tr, 'n_val': n_val, 'n_te': n_te,
        'size': size, 'bs': bs
    }
    return dl_tr, dl_val, dl_te, info


# Create single data loader for entire dataset
def make_loader(path, size, bs, shuf=False, aug=False, wk=4):
    tf = get_trans(size, aug=aug)
    ds = ImgDS(path, transform=tf)
    print(f"N: {len(ds)}, C: {ds.get_n_classes()}")

    # Create data loader without splitting
    dl = DataLoader(ds, batch_size=bs, shuffle=shuf, num_workers=wk, pin_memory=False)
    info = {
        'n_cls': ds.get_n_classes(), 'classes': ds.get_classes(),
        'n_tot': len(ds), 'size': size, 'bs': bs
    }
    return dl, info


# Visualize sample images from data loader
def plot_samples(dl, n, out, names):
    import matplotlib.pyplot as plt
    try:
        # Get first batch from data loader
        imgs, lbls = next(iter(dl))
    except Exception as e:
        print(f"Viz err: {e}")
        return

    # Take first n samples
    imgs = imgs[:n]
    lbls = lbls[:n]

    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    imgs = imgs * std + mean
    imgs = torch.clamp(imgs, 0, 1)

    # Calculate subplot grid size
    nc = min(4, n)
    nr = (n + nc - 1) // nc
    if nr == 0: nr = 1

    # Create figure and plot images
    fig, ax = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3))
    if n == 1: ax = np.array([ax])
    ax = ax.flatten()

    # Plot each image with its label
    for i, (im, lb) in enumerate(zip(imgs, lbls)):
        if i >= len(ax): break
        im_np = im.cpu().numpy().transpose(1, 2, 0)
        ax[i].imshow(im_np)
        if lb < len(names):
            ax[i].set_title(f'{names[lb]}')
        else:
            ax[i].set_title(f'C {lb}')
        ax[i].axis('off')

    # Hide unused subplots
    for i in range(len(imgs), len(ax)): ax[i].axis('off')

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

