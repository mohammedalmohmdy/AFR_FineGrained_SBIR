
import torch, torch.nn as nn
import torchvision.models as tv

class SpatialEncoder(nn.Module):
    def __init__(self, name="resnet50"):
        super().__init__()
        net = getattr(tv, name)(weights=tv.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(net.children())[:-2])
        self.out_dim = 2048

    def forward(self, x):
        return self.features(x)
