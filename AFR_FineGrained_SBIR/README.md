
# AFR-SBIR: Adaptive Frequency Reasoning for Fine-Grained Sketch-Based Image Retrieval

This repository provides a reference PyTorch implementation of **AFR-SBIR**, an instance-adaptive framework for FG-SBIR.
The codebase is structured to be reproducible, extensible, and aligned with the methodology described in the paper.

## Features
- Shared spatial encoder for sketch/photo
- Instance-adaptive frequency hypothesis reasoning
- Adaptive frequency routing
- Confidence-aware spatial–frequency fusion
- Metric embedding learning with triplet loss
- Support for Sketchy, TU-Berlin, QMUL-Chair, QMUL-Shoe-V2

## Environment
- Python >= 3.9
- PyTorch >= 2.0
- torchvision, numpy, pillow, tqdm

## Quick Start
```bash
pip install -r requirements.txt
python train.py --config configs/afrsbir_qmul_shoe.yaml
```

## Reproducibility
- Deterministic seeds supported
- Config-driven experiments
- Checkpoints and logs saved automatically

## Disclaimer
This is a research reference implementation. Dataset paths must be set locally.
