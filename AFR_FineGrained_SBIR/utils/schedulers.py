
import torch

def build_scheduler(optimizer, name="step"):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
