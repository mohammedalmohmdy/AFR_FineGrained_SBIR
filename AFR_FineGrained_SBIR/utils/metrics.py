"""Retrieval metrics (Section 4.2 / Equations 22, CMC).

All metrics are computed against an explicit positive mask ``pos_mask`` of shape
(Q, G) (1 when gallery item j is a positive match for query i). This is the
correct benchmark protocol for both instance-level (single positive per query)
and category-level (all same-category gallery items are positive) settings; it
never assumes query count == gallery count and never uses diagonal matching.
"""

import torch


def pairwise_distance(a, b):
    """Euclidean distance matrix between rows of a (Q, D) and b (G, D)."""
    return torch.cdist(a, b, p=2)


def rank_positive_mask(dist, pos_mask):
    """Return a (Q, G) boolean tensor: row i, col r = True if the r-th ranked
    gallery item is positive for query i."""
    order = dist.argsort(dim=1)                    # (Q, G)
    return torch.gather(pos_mask, 1, order)        # hits in ranked order


def topk_accuracy(dist, pos_mask, k):
    """Fraction of queries with >=1 positive in the top-k ranked gallery."""
    ranked = rank_positive_mask(dist, pos_mask)
    return ranked[:, :k].any(dim=1).float().mean().item()


def average_precision(dist, pos_mask):
    """mAP over queries (Eq. 22)."""
    ranked = rank_positive_mask(dist, pos_mask)    # (Q, G) bool, ranked order
    num_pos = pos_mask.sum(dim=1).clamp(min=1)     # (Q,)
    ranks = torch.arange(1, ranked.shape[1] + 1, device=dist.device,
                         dtype=torch.float32)
    # precision at each rank position, then weight by the relevance flag.
    prec = ranked.cumsum(dim=1).float() / ranks    # (Q, G)
    ap = (prec * ranked.float()).sum(dim=1) / num_pos
    return ap.mean().item()


def precision_at_k(dist, pos_mask, k):
    """P@k: fraction of the top-k retrieved items that are positive (mean over
    queries)."""
    ranked = rank_positive_mask(dist, pos_mask)
    return ranked[:, :k].float().mean().item()


def cmc_curve(dist, pos_mask, max_rank=None):
    """Cumulative matching characteristic: at rank K the fraction of queries
    whose (first) positive is within the top-K ranked gallery."""
    ranked = rank_positive_mask(dist, pos_mask)
    max_rank = max_rank or ranked.shape[1]
    hits = ranked[:, :max_rank].any(dim=1).float()      # (Q,)
    cum = ranked.cumsum(dim=1)[:, :max_rank]            # (Q, G) count of positives up to K
    # "correct positive retrieved within top-K" -> any-positive within K
    hit_at_k = (cum >= 1).float().mean(dim=0)           # (max_rank,)
    return hit_at_k


def evaluate_retrieval(dist, pos_mask, top_ks=(1, 5, 10, 20), p_at=100,
                       protocol="instance"):
    """Return a dict of metric name -> value for the given benchmark protocol.

    category-level  -> mAP, P@100 (+ Top-K for completeness)
    instance-level  -> Top-K, mAP, CMC
    """
    res = {}
    for k in top_ks:
        res[f"Top-{k}"] = topk_accuracy(dist, pos_mask, k)
    res["mAP"] = average_precision(dist, pos_mask)
    res[f"P@{p_at}"] = precision_at_k(dist, pos_mask, p_at)

    cmc = cmc_curve(dist, pos_mask)
    res["CMC"] = {k: cmc[k - 1].item() if k - 1 < cmc.numel() else
                  (cmc[-1].item() if cmc.numel() else 0.0) for k in top_ks}
    return res


def build_pos_mask(query_labels, gallery_labels):
    """(Q,) vs (G,) int label tensors -> (Q, G) bool positive mask."""
    return query_labels.view(-1, 1) == gallery_labels.view(1, -1)