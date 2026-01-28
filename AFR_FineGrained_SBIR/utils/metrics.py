
import torch
def pairwise_distance(a, b):
    return torch.cdist(a, b, p=2)

def topk_accuracy(dist, k=1):
    # dist: [N, N], smaller is better
    idx = dist.argsort(dim=1)
    correct = torch.arange(dist.size(0), device=dist.device)
    return (idx[:, :k] == correct.unsqueeze(1)).any(dim=1).float().mean().item()
