from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models
import timm


class FeatureExtractor(nn.Module):
    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.model = None
        self.feature_dim = None
        self.input_size = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def get_feature_dim(self) -> int:
        return self.feature_dim
    
    def get_input_size(self) -> int:
        return self.input_size


class VGG16Extractor(FeatureExtractor):
    def __init__(self, device: torch.device):
        super().__init__('vgg16', device)
        
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        self.features = self.model.features
        self.avgpool = self.model.avgpool
        self.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        
        self.model = nn.Sequential(self.features, self.avgpool, nn.Flatten(), self.classifier)
        self.model.to(device)
        self.model.eval()
        
        self.feature_dim = 4096
        self.input_size = 224
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model(x)
        return features


class ResNet50Extractor(FeatureExtractor):
    def __init__(self, device: torch.device):
        super().__init__('resnet50', device)
        
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(device)
        self.model.eval()
        
        self.feature_dim = 2048
        self.input_size = 224
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model(x)
            features = features.flatten(1)
        return features


class DenseNet121Extractor(FeatureExtractor):
    def __init__(self, device: torch.device):
        super().__init__('densenet121', device)
        
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        
        self.features = self.model.features
        self.model = nn.Sequential(
            self.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.model.to(device)
        self.model.eval()
        
        self.feature_dim = 1024
        self.input_size = 224
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model(x)
        return features


class DinoV2Extractor(FeatureExtractor):
    def __init__(self, device: torch.device, model_size: str = 'base'):
        super().__init__(f'dinov2_{model_size}', device)
        
        model_name = f'vit_{model_size}_patch14_dinov2.lvd142m'
        
        try:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load DinoV2: {e}")
        
        feature_dims = {
            'small': 384,
            'base': 768,
            'large': 1024,
            'giant': 1536
        }
        
        self.feature_dim = feature_dims.get(model_size, 768)
        self.input_size = 518
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model(x)
        return features


def create_feature_extractor(model_name: str, device: torch.device) -> FeatureExtractor:
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'vgg16':
        extractor = VGG16Extractor(device)
    elif model_name_lower == 'resnet50':
        extractor = ResNet50Extractor(device)
    elif model_name_lower == 'densenet121':
        extractor = DenseNet121Extractor(device)
    elif model_name_lower.startswith('dinov2'):
        if '_' in model_name_lower:
            size = model_name_lower.split('_')[1]
        else:
            size = 'base'
        extractor = DinoV2Extractor(device, model_size=size)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Feature dim: {extractor.get_feature_dim()}, Input size: {extractor.get_input_size()}")
    
    return extractor


def get_model_info(model_name: str) -> Tuple[int, int]:
    model_configs = {
        'vgg16': (4096, 224),
        'resnet50': (2048, 224),
        'densenet121': (1024, 224),
        'dinov2': (768, 518),
        'dinov2_small': (384, 518),
        'dinov2_base': (768, 518),
        'dinov2_large': (1024, 518),
        'dinov2_giant': (1536, 518)
    }
    
    model_name_lower = model_name.lower()
    if model_name_lower not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_configs[model_name_lower]
