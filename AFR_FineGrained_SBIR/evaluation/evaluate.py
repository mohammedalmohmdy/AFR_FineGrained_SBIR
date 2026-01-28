
import torch
from utils.metrics import pairwise_distance, topk_accuracy

def evaluate(emb_s, emb_p, ks=(1,5,10,20)):
    dist = pairwise_distance(emb_s, emb_p)
    results = {}
    for k in ks:
        results[f"Top-{k}"] = topk_accuracy(dist, k)
    return results
