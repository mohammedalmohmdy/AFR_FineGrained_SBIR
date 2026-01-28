
import torch, torch.nn as nn
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    def forward(self, zs, zp_pos, zp_neg):
        d_pos = (zs - zp_pos).pow(2).sum(1)
        d_neg = (zs - zp_neg).pow(2).sum(1)
        return torch.clamp(d_pos - d_neg + self.margin, min=0).mean()
