"""Optional LR schedulers.

The paper specifies Adam with a constant learning rate of 1e-4 for 100 epochs
and does not describe a learning-rate schedule, so the default is ``none``.
"""

import torch


def build_scheduler(optimizer, name="none", **kwargs):
    name = (name or "none").lower()
    if name in ("none", "constant", "const"):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs.get("t_max", 100))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get("step_size", 20),
            gamma=kwargs.get("gamma", 0.1))
    raise ValueError(f"Unknown scheduler '{name}'")