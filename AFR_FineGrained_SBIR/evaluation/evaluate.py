"""Retrieval evaluation (Section 4.2).

Builds L2-normalised embeddings for the query (sketch) and gallery (photo)
sets, computes the query x gallery Euclidean distance matrix, builds the
benchmark-specific positive mask from the protocol labels (category for
category-level, instance for instance-level), and reports Top-1/5/10/20, mAP,
P@100 and CMC. It never assumes query count == gallery count.
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import pairwise_distance, evaluate_retrieval, build_pos_mask
from utils.config import load_config, resolve_device
from models.afrsbir import build_model


@torch.no_grad()
def embed_dataset(model, dataset, device, batch_size=32, num_workers=0):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    embs, labels = [], []
    for img, lab in loader:
        embs.append(model.encode(img.to(device)).cpu())
        labels.append(lab)
    return torch.cat(embs, dim=0), torch.cat(labels, dim=0)


def run_evaluation(model, query_ds, gallery_ds, protocol, device,
                   batch_size=32, top_ks=(1, 5, 10, 20), p_at=100,
                   num_workers=0):
    q_emb, q_lab = embed_dataset(model, query_ds, device, batch_size, num_workers)
    g_emb, g_lab = embed_dataset(model, gallery_ds, device, batch_size, num_workers)

    dist = pairwise_distance(q_emb.to(device), g_emb.to(device))
    pos_mask = build_pos_mask(q_lab.to(device), g_lab.to(device))
    metrics = evaluate_retrieval(dist, pos_mask, top_ks=top_ks, p_at=p_at,
                                 protocol=protocol)
    return metrics, {"num_queries": len(q_lab), "num_gallery": len(g_lab)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="AFR-SBIR retrieval evaluation")
    parser.add_argument("--config", required=True, help="Path to dataset/model YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(cfg)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Instantiate transforms lazily (avoids importing torchvision at module import).
    from utils.transforms import build_transform
    tfm = build_transform(cfg)

    from data.datasets import build_datasets
    _, query_ds, gallery_ds, protocol, stats = build_datasets(cfg, tfm)

    if args.batch_size is None:
        args.batch_size = cfg.get("training", {}).get("batch_size", 32)

    metrics, info = run_evaluation(model, query_ds, gallery_ds, protocol,
                                   device, batch_size=args.batch_size,
                                   num_workers=cfg.get("dataloader", {}).get("num_workers", 0))

    print(f"[evaluate] queries={info['num_queries']} gallery={info['num_gallery']} "
          f"protocol={protocol}")
    for k, v in metrics.items():
        if k == "CMC":
            print(f"  {k}: " + ", ".join(f"r{rank}={val:.4f}" for rank, val in v.items()))
        else:
            print(f"  {k}={v:.4f}")

    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    main()