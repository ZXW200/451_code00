from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np



class ImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        
        self.samples = []
        self.class_names = []
        self.class_to_idx = {}
        
        self._load_dataset()
        
    def _load_dataset(self):
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        if not class_dirs:
            raise ValueError(f"No subdirectories found in {self.root_dir}")
        
        self.class_names = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            
            for ext in self.extensions:
                for img_path in class_dir.glob(f'*{ext}'):
                    self.samples.append((str(img_path), class_idx))
        
        if not self.samples:
            raise ValueError(f"No images found in {self.root_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self) -> List[str]:
        return self.class_names
    
    def get_num_classes(self) -> int:
        return len(self.class_names)


def get_transforms(input_size: int = 224, augment: bool = False) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augment:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    return transform



def create_dataloaders(
    dataset_path: str,
    input_size: int,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    print(f"Loading dataset: {dataset_path}")
    
    transform = get_transforms(input_size, augment=False)
    full_dataset = ImageDataset(dataset_path, transform=transform)
    
    print(f"Samples: {len(full_dataset)}, Classes: {full_dataset.get_num_classes()}")
    
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_transform = get_transforms(input_size, augment=True)
    train_dataset.dataset.transform = train_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataset.dataset.transform = get_transforms(input_size, augment=False)
    test_dataset.dataset.transform = get_transforms(input_size, augment=False)
    
    dataset_info = {
        'num_classes': full_dataset.get_num_classes(),
        'class_names': full_dataset.get_class_names(),
        'total_samples': total_size,
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': test_size,
        'input_size': input_size,
        'batch_size': batch_size
    }
    
    return train_loader, val_loader, test_loader, dataset_info


def create_single_dataloader(
    dataset_path: str,
    input_size: int,
    batch_size: int,
    shuffle: bool = False,
    augment: bool = False,
    num_workers: int = 4
) -> Tuple[DataLoader, Dict[str, Any]]:
    transform = get_transforms(input_size, augment=augment)
    dataset = ImageDataset(dataset_path, transform=transform)
    
    print(f"Samples: {len(dataset)}, Classes: {dataset.get_num_classes()}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataset_info = {
        'num_classes': dataset.get_num_classes(),
        'class_names': dataset.get_class_names(),
        'total_samples': len(dataset),
        'input_size': input_size,
        'batch_size': batch_size
    }
    
    return dataloader, dataset_info


def visualize_samples(
    dataloader: DataLoader,
    num_samples: int,
    output_path: Path,
    class_names: List[str]
):
    import matplotlib.pyplot as plt
    
    images, labels = next(iter(dataloader))
    
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    n_cols = min(4, num_samples)
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        if idx >= len(axes):
            break
        
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        
        axes[idx].imshow(img_np)
        axes[idx].set_title(f'Class: {class_names[label]}')
        axes[idx].axis('off')
    
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
