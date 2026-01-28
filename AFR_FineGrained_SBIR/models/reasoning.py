
import torch, torch.nn as nn, torch.nn.functional as F

class InstanceReasoner(nn.Module):
    def __init__(self, in_channels, k):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, k)
    def forward(self, x):
        v = self.pool(x).flatten(1)
        return F.softmax(self.fc(v), dim=1)

class ConfidenceEstimator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1)
    def forward(self, x):
        v = self.pool(x).flatten(1)
        return torch.sigmoid(self.fc(v))
