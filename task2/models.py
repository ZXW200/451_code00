import torch
import torch.nn as nn
from torchvision import models
import timm


class BaseExt(nn.Module):
    def __init__(self, name, dev):
        super().__init__()
        self.name = name
        self.dev = dev
        self.net = None
        self.dim = None
        self.size = None

    def forward(self, x):
        raise NotImplementedError

    def get_dim(self):
        return self.dim

    def get_size(self):
        return self.size


class ResNetExt(BaseExt):
    def __init__(self, dev):
        super().__init__('resnet50', dev)

        self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.net = nn.Sequential(*list(self.net.children())[:-1])
        self.net.to(dev)
        self.net.eval()

        self.dim = 2048
        self.size = 224

    def forward(self, x):
        with torch.no_grad():
            f = self.net(x)
            f = f.flatten(1)
        return f


class DenseNetExt(BaseExt):
    def __init__(self, dev):
        super().__init__('densenet121', dev)

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

        self.dim = 1024
        self.size = 224

    def forward(self, x):
        with torch.no_grad():
            f = self.net(x)
        return f


class DinoExt(BaseExt):
    def __init__(self, dev, size='base'):
        super().__init__(f'dinov2_{size}', dev)

        m_name = f'vit_{size}_patch14_dinov2.lvd142m'

        try:
            self.net = timm.create_model(m_name, pretrained=True, num_classes=0)
            self.net.to(dev)
            self.net.eval()
        except Exception as e:
            raise RuntimeError(f"Dino err: {e}")

        dims = {'small': 384, 'base': 768, 'large': 1024, 'giant': 1536}
        self.dim = dims.get(size, 768)
        self.size = 518

    def forward(self, x):
        with torch.no_grad():
            f = self.net(x)
        return f


def get_ext(name, dev):
    name = name.lower()

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
