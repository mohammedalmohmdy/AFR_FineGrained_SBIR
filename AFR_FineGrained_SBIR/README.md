
# AFR-SBIR: Adaptive Frequency Reasoning for Fine-Grained Sketch-Based Image Retrieval

This repository provides a reference PyTorch implementation of **AFR-SBIR**, an instance-adaptive framework for FG-SBIR.


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

Datasets
ShoeV2 / ChairV2
Sketchy Official Website
Google Drive Download

Sketchy
Sketchy Official Website
Google Drive Download

TU-Berlin
TU-Berlin Official Website
Google Drive Download

### 📂 Datasets

- **ShoeV2 / ChairV2**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/1frltfiEd9ymnODZFHYrbg741kfys1rq1/view)

- **Sketchy**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view)

- **TU-Berlin**  
  [TU-Berlin Official Website](https://www.tu-berlin.de/)  
  [Google Drive Download](https://drive.google.com/file/d/12VV40j5Nf4hNBfFy0AhYEtql1OjwXAUC/view)


  Citation: If you use this code, please cite:

title = {AFR-SBIR: Adaptive Frequency Reasoning for Fine-Grained Sketch-Based Image Retrieval},

author = {Mohammed A. S. Al-Mohamadi and Prabhakar C. J.},

journal = {international journal of machine learning and cybernetics}, year = {2026} }

Contact: almohmdy30@gmail.com GitHub: https://github.com/mohammedalmohmdy
