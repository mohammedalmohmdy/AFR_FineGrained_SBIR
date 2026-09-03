"""Training objective (Section 3.9).

    L_tri   = 1/N sum_i [ d(f_s^i, f_p^{i,+}) - d(f_s^i, f_p^{i,-}) + m ]_+
    L_conf  = 1/N sum_i (c_i - 0.5)^2
    L_total = L_tri + lambda * L_conf

The paper fixes the triplet margin m = 0.5 and the confidence weight
lambda = 0.10. ``d(.,.)`` is the squared Euclidean distance between L2-normalised
embeddings. Negative photos are mined hard-in-batch: for anchor i the farthest
negative is the gallery photo of a *different* instance/category (i.e. a photo
whose pair label differs from the anchor's label).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, zs, zp, labels):
        """zs, zp: (B, D) L2-normalised embeddings. labels: (B,) pair/instance ids."""
        B = zs.shape[0]
        dist = torch.cdist(zs, zp, p=2).pow(2)          # squared Euclidean
        pos = dist.diagonal()                            # matched positive

        labels = labels.view(1, -1)
        neg_mask = labels != labels.view(-1, 1)          # (B, B), exclude same-pair
        d_neg = dist.clone()
        if neg_mask.any():
            d_neg = d_neg.masked_fill(~neg_mask, float("inf"))
            hardest = d_neg.min(dim=1).values
        else:
            hardest = pos.detach().clone()               # no negatives available

        loss = F.relu(pos - hardest + self.margin)
        return loss.mean()


class ConfidenceLoss(nn.Module):
    """L_conf = mean((c - 0.5)^2) (Eq. 21)."""

    def forward(self, confidence):
        return ((confidence - 0.5) ** 2).mean()


class TotalLoss(nn.Module):
    """L_total = L_tri + lambda * L_conf (Eq. 20)."""

    def __init__(self, margin=0.5, lambda_conf=0.1):
        super().__init__()
        self.triplet = TripletLoss(margin=margin)
        self.conf = ConfidenceLoss()
        self.lambda_conf = float(lambda_conf)

    def forward(self, zs, zp, labels, confidence):
        l_tri = self.triplet(zs, zp, labels)
        # Ablation variants without the confidence pathway (Table 6) receive
        # confidence=None and contribute no regularisation term.
        if confidence is None:
            return l_tri, l_tri, confidence
        l_conf = self.conf(confidence)
        total = l_tri + self.lambda_conf * l_conf
        return total, l_tri, l_conf