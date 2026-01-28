
import torch, torch.nn as nn, torch.nn.functional as F
from .backbone import SpatialEncoder
from .frequency import FrequencyBank
from .reasoning import InstanceReasoner, ConfidenceEstimator

class AFRSBIR(nn.Module):
    def __init__(self, backbone="resnet50", embed_dim=256, k=3):
        super().__init__()
        self.encoder = SpatialEncoder(backbone)
        C = self.encoder.out_dim
        self.freq_bank = FrequencyBank(C)
        self.reasoner = InstanceReasoner(C, k)
        self.conf = ConfidenceEstimator(C)
        self.proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(C, embed_dim))

    def forward(self, s, p):
        Fs, Fp = self.encoder(s), self.encoder(p)
        alphas = self.reasoner(Fs)                # sketch-driven
        Fs_k = self.freq_bank(Fs)
        Fp_k = self.freq_bank(Fp)
        Fs_freq = sum(alphas[:,i].view(-1,1,1,1)*Fs_k[i] for i in range(len(Fs_k)))
        Fp_freq = sum(alphas[:,i].view(-1,1,1,1)*Fp_k[i] for i in range(len(Fp_k)))
        c = self.conf(Fs)
        Fs_final = c*Fs_freq + (1-c)*Fs
        Fp_final = c*Fp_freq + (1-c)*Fp
        zs = F.normalize(self.proj(Fs_final), dim=1)
        zp = F.normalize(self.proj(Fp_final), dim=1)
        return zs, zp, alphas, c
