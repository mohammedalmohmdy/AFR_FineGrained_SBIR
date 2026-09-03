"""Configuration loading and small helpers shared by train/eval/experiments.

The CLI contract is ``--config path/to.yaml`` (with optional ``--opts``
key=value overrides). ``load_config`` merges user config onto ``base_defaults``
so that a minimal YAML is sufficient while the paper's constants are preserved.
"""

import os

import yaml


BASE_DEFAULTS = {
    "experiment": {"name": "AFR-SBIR", "seed": 42, "device": "auto"},
    "dataset": {"name": "QMUL-Shoe-V2", "root": "/path/to/qmul_shoe",
                "image_size": 224},
    "model": {"backbone": "resnet50", "embedding_dim": 256,
              "num_freq_hypotheses": 3, "frequency_hidden": 512,
              "confidence_hidden": 256, "use_frequency": True,
              "weight_mode": "adaptive", "use_confidence": True,
              "feature_norm": True, "pretrained": True},
    "training": {"batch_size": 32, "epochs": 100, "lr": 1e-4,
                 "margin": 0.5, "lambda_conf": 0.1, "optimizer": "adam",
                 "weight_decay": 0.0, "scheduler": "none",
                 "save_every": 10, "eval_every": 10, "resume": None},
    "evaluation": {"top_ks": [1, 5, 10, 20], "p_at": 100},
    "dataloader": {"num_workers": 0},
}


def _deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path, overrides=None):
    """Load a YAML config and merge it onto the paper's base defaults."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _deep_merge(BASE_DEFAULTS, raw)
    # Apply CLI overrides of the form section.key=value (dotted keys supported).
    for ov in (overrides or []):
        if "=" not in ov:
            raise ValueError(f"invalid --opts item {ov!r} (expected key=value)")
        key, val = ov.split("=", 1)
        cursor = cfg
        keys = key.split(".")
        for k in keys[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[keys[-1]] = _coerce(val)
    cfg["experiment"]["seed"] = int(cfg["experiment"]["seed"])
    return cfg


def _coerce(s):
    s = s.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # comma separated list -> list of floats/ints
    if "," in s:
        return [_coerce(x) for x in s.split(",")]
    return s


def resolve_device(cfg):
    dev = cfg.get("experiment", {}).get("device", "auto")
    if dev == "auto":
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import torch
    return torch.device(dev)


def output_dir(cfg, base="./runs"):
    name = cfg["experiment"].get("name", "AFR-SBIR")
    seed = cfg["experiment"].get("seed", 42)
    out = cfg["experiment"].get("output_dir", os.path.join(base, f"{name}_seed{seed}"))
    os.makedirs(out, exist_ok=True)
    return out