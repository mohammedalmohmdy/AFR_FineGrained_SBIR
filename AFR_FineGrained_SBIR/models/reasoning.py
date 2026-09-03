"""Instance-adaptive reasoning and confidence estimation (Sections 3.5 and 3.7).

InstanceAdaptiveFrequencyReasoning
    vs = GAP(F_s)                      -> R^C
    hidden = ReLU(Linear(C, 512))      -> R^512
    logits = Linear(512, K)            -> R^K
    alpha  = Softmax(logits)           -> R^K

ConfidenceEstimator
    hs = GAP(F_s)
    hc = ReLU(Linear(C, 256))          -> R^256
    c  = Sigmoid(Linear(256, 1))       -> [0, 1]

Both modules are driven exclusively by the *sketch* representation, as stated
in the manuscript. The softmax frequency weights produced by the reasoning
module are then shared symmetrically with the photo branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceReasoner(nn.Module):
    """Instance-adaptive frequency weighting (Eqs. 7-9).

    GAP -> Linear(C, hidden) -> ReLU -> Linear(hidden, K) -> Softmax.
    The paper fixes the hidden dimension at 512 and K at 3.
    """

    def __init__(self, in_channels, k=3, hidden=512):
        super().__init__()
        self.k = k
        self.hidden = hidden
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, k),
        )

    def forward(self, x):
        v = self.pool(x).flatten(1)          # (B, C)
        logits = self.mlp(v)                 # (B, K)
        alphas = F.softmax(logits, dim=1)    # (B, K)
        return alphas, logits


class ConfidenceEstimator(nn.Module):
    """Confidence estimation network (Eqs. 12-14).

    GAP -> Linear(C, hidden) -> ReLU -> Linear(hidden, 1) -> Sigmoid.
    The paper fixes the hidden dimension at 256.
    """

    def __init__(self, in_channels, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        v = self.pool(x).flatten(1)          # (B, C)
        c = torch.sigmoid(self.mlp(v))       # (B, 1)
        return c