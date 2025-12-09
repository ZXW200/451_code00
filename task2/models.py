import torch
import torch.nn as nn
from torchvision import models
import timm


# Base class for all feature extractors
class BaseExt(nn.Module):
    def __init__(self, name, dev):
        super().__init__()
        self.name = name
        self.dev = dev
        self.net = None
        self.dim = None
        self.size = None

    # Forward pass must be implemented by subclasses
    def forward(self, x):
        raise NotImplementedError

    # Return feature dimension of the model
    def get_dim(self):
        return self.dim

    # Return required input image size
    def get_size(self):
        return self.size


# ResNet50 feature extractor using ImageNet pretrained weights
class ResNetExt(BaseExt):
    def __init__(self, dev):
        super().__init__('resnet50', dev)

        # Load pretrained ResNet50 and remove classification head
        self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.net = nn.Sequential(*list(self.net.children())[:-1])
        self.net.to(dev)
        self.net.eval()

        # ResNet50 outputs 2048-dim features at 224x224 input
        self.dim = 2048
        self.size = 224

    # Extract features without gradients
    def forward(self, x):
        with torch.no_grad():
            f = self.net(x)
            f = f.flatten(1)
        return f


# DenseNet121 feature extractor using ImageNet pretrained weights
class DenseNetExt(BaseExt):
    def __init__(self, dev):
        super().__init__('densenet121', dev)

        # Load pretrained DenseNet121 and extract features before classifier
        self.net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.feats = self.net.features
        self.net = nn.Sequential(
            self.feats,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.net.to(dev)
        self.net.eval()

        # DenseNet121 outputs 1024-dim features at 224x224 input
        self.dim = 1024
        self.size = 224

    # Extract features without gradients
    def forward(self, x):
        with torch.no_grad():
            f = self.net(x)
        return f


# DINOv2 vision transformer feature extractor
class DinoExt(BaseExt):
    def __init__(self, dev, size='base'):
        super().__init__(f'dinov2_{size}', dev)

        # Load DINOv2 model from timm library
        m_name = f'vit_{size}_patch14_dinov2.lvd142m'

        try:
            self.net = timm.create_model(m_name, pretrained=True, num_classes=0)
            self.net.to(dev)
            self.net.eval()
        except Exception as e:
            raise RuntimeError(f"Dino err: {e}")

        # Different sizes have different feature dimensions
        dims = {'small': 384, 'base': 768, 'large': 1024, 'giant': 1536}
        self.dim = dims.get(size, 768)
        self.size = 518

    # Extract features without gradients
    def forward(self, x):
        with torch.no_grad():
            f = self.net(x)
        return f


# Factory function to create feature extractor by name
def get_ext(name, dev):
    name = name.lower()

    # Select and instantiate appropriate extractor
    if name == 'resnet50':
        ext = ResNetExt(dev)
    elif name == 'densenet121':
        ext = DenseNetExt(dev)
    elif name.startswith('dinov2'):
        size = name.split('_')[1] if '_' in name else 'base'
        ext = DinoExt(dev, size=size)
    else:
        raise ValueError(f"Unknown: {name}")

    print(f"Model: {ext.get_dim()} dim, {ext.get_size()} size")
    return ext

