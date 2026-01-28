
import torch, torch.nn as nn

class FrequencyHypothesis(nn.Module):
    def __init__(self, in_channels, mode="coarse"):
        super().__init__()
        k = {"coarse":1, "mid":3, "fine":5}[mode]
        self.conv = nn.Conv2d(in_channels, in_channels, k, padding=k//2, groups=in_channels)
    def forward(self, x):
        return self.conv(x)

class FrequencyBank(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.modes = ["coarse","mid","fine"]
        self.bank = nn.ModuleList([FrequencyHypothesis(in_channels, m) for m in self.modes])
    def forward(self, x):
        return [h(x) for h in self.bank]
