"""Shared-weight spatial encoder (Section 3.3).

Both sketch and photo inputs are processed by the same ResNet-50 backbone
pretrained on ImageNet, producing C=2048 channel feature maps
(F_s, F_p in R^{C x H x W}). The final classification layers are discarded; the
output is the feature map before global pooling.
"""

import torch
import torch.nn as nn
import torchvision.models as tv

_WEIGHTS = {
    "resnet50": tv.ResNet50_Weights.IMAGENET1K_V1,
    "resnet34": tv.ResNet34_Weights.IMAGENET1K_V1,
    "resnet18": tv.ResNet18_Weights.IMAGENET1K_V1,
}


class SpatialEncoder(nn.Module):
    def __init__(self, name="resnet50", pretrained=True):
        super().__init__()
        self.name = name
        if name not in _WEIGHTS:
            raise ValueError(f"Unsupported backbone '{name}'. Available: {list(_WEIGHTS)}")
        weights = _WEIGHTS[name] if pretrained else None
        net = getattr(tv, name)(weights=weights)
        # Keep everything except the final adaptive-pool and fc layers.
        self.features = nn.Sequential(*list(net.children())[:-2])
        self.out_dim = 2048 if name == "resnet50" else 512

    def forward(self, x):
        return self.features(x)