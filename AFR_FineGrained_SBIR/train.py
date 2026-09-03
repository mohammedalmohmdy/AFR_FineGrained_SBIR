"""AFR-SBIR training entry point.

Usage:
    python train.py --config configs/afrsbir_qmul_shoe.yaml
    python train.py --config configs/afrsbir_qmul_shoe.yaml --opts training.epochs=5

Trains the full AFR-SBIR model with the objective

    L_total = L_triplet + 0.10 * L_conf   (margin 0.5, lambda_conf 0.1)

and logs all three terms plus validation metrics each ``eval_every`` epochs.
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_config, resolve_device, output_dir
from utils.seed import set_seed
from utils.transforms import build_transform
from utils.schedulers import build_scheduler
from data.datasets import build_datasets
from models.afrsbir import build_model
from losses.triplet import TotalLoss


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train AFR-SBIR")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--opts", nargs="*", default=[], metavar="key=value",
                   help="Override config entries, e.g. training.epochs=5")
    return p.parse_args(argv)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    totals, tri_sum, conf_sum = 0.0, 0.0, 0.0
    n = 0
    for s, p, photo_id in loader:
        s = s.to(device)
        p = p.to(device)
        photo_id = photo_id.to(device)

        out = model(s, p)
        total, l_tri, l_conf = criterion(out["zs"], out["zp"], photo_id,
                                         out["confidence"])

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        totals += total.item()
        tri_sum += l_tri.item()
        conf_sum += l_conf.item()
        n += 1
    return totals / n, tri_sum / n, conf_sum / n


def run_eval(model, query_ds, gallery_ds, protocol, device, cfg):
    if len(query_ds) == 0 or len(gallery_ds) == 0:
        return None
    from evaluation.evaluate import run_evaluation
    metrics, _ = run_evaluation(
        model, query_ds, gallery_ds, protocol, device,
        batch_size=cfg["training"]["batch_size"],
        top_ks=tuple(cfg["evaluation"]["top_ks"]),
        p_at=cfg["evaluation"]["p_at"],
        num_workers=cfg["dataloader"]["num_workers"],
    )
    return metrics


def save_checkpoint(path, model, optimizer, epoch, cfg, metrics=None):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "metrics": metrics,
    }, path)


def main(argv=None):
    args = _parse_args(argv)
    cfg = load_config(args.config, args.opts)
    device = resolve_device(cfg)
    set_seed(cfg["experiment"]["seed"])

    outdir = output_dir(cfg)
    print(f"[train] output dir: {outdir}")
    print(f"[train] device: {device}")

    tfm = build_transform(cfg)
    train_ds, query_ds, gallery_ds, protocol, stats = build_datasets(cfg, tfm)
    print(f"[train] dataset split: {stats}")

    tr_cfg = cfg["training"]
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg.get("weight_decay", 0.0))
    scheduler = build_scheduler(optimizer, tr_cfg.get("scheduler", "none"),
                                t_max=tr_cfg["epochs"])
    criterion = TotalLoss(margin=tr_cfg["margin"], lambda_conf=tr_cfg["lambda_conf"])
    print(f"[train] margin={tr_cfg['margin']} lambda_conf={tr_cfg['lambda_conf']} "
          f"lr={tr_cfg['lr']} batch_size={tr_cfg['batch_size']} epochs={tr_cfg['epochs']}")

    start_epoch = 0
    if tr_cfg.get("resume") and os.path.isfile(tr_cfg["resume"]):
        ckpt = torch.load(tr_cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[train] resumed from {tr_cfg['resume']} at epoch {start_epoch}")

    # Log header + per-epoch lines to a CSV for the convergence figure (App. A).
    log_path = os.path.join(outdir, "train.log")
    with open(log_path, "w") as f:
        f.write("epoch,l_total,l_triplet,l_conf")
        if query_ds is not None:
            f.write(",mAP,Top-1")
        f.write("\n")

    train_loader = None
    if train_ds is not None and len(train_ds) > 0:
        train_loader = DataLoader(train_ds, batch_size=tr_cfg["batch_size"],
                                  shuffle=True, drop_last=True,
                                  num_workers=cfg["dataloader"]["num_workers"])

    best = {"mAP": -1.0}
    for epoch in range(start_epoch, tr_cfg["epochs"]):
        t0 = time.time()
        if train_loader is None:
            print(f"[train] no training pairs found; skipping epoch {epoch}")
            continue

        l_total, l_tri, l_conf = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        if scheduler is not None:
            scheduler.step()

        metrics = None
        if (epoch + 1) % tr_cfg.get("eval_every", 10) == 0:
            metrics = run_eval(model, query_ds, gallery_ds, protocol, device, cfg)

        msg = (f"[train] epoch {epoch + 1}/{tr_cfg['epochs']} "
               f"L_total={l_total:.5f} L_triplet={l_tri:.5f} L_conf={l_conf:.5f} "
               f"({time.time() - t0:.1f}s)")
        if metrics:
            msg += f" mAP={metrics['mAP']:.4f} Top-1={metrics['Top-1']:.4f}"
        print(msg)

        with open(log_path, "a") as f:
            f.write(f"{epoch + 1},{l_total:.6f},{l_tri:.6f},{l_conf:.6f}")
            if metrics:
                f.write(f",{metrics['mAP']:.6f},{metrics['Top-1']:.6f}")
            f.write("\n")

        if (epoch + 1) % tr_cfg.get("save_every", 10) == 0:
            save_checkpoint(os.path.join(outdir, f"checkpoint_epoch{epoch + 1}.pt"),
                            model, optimizer, epoch, cfg, metrics)

        if metrics and metrics["mAP"] > best["mAP"]:
            best["mAP"] = metrics["mAP"]
            save_checkpoint(os.path.join(outdir, "best.pt"),
                            model, optimizer, epoch, cfg, metrics)

    save_checkpoint(os.path.join(outdir, "last.pt"), model, optimizer, epoch, cfg, metrics)
    print(f"[train] done. best mAP={best['mAP']:.4f}  checkpoints in {outdir}")


if __name__ == "__main__":
    main()