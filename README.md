# Grapevine Downy Mildew Severity Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20701850.svg)](https://doi.org/10.5281/zenodo.20701850)

> **Explainable Deep Learning for Field-Deployable Grapevine Downy Mildew Severity Classification: A Comparative Analysis with Edge-Deployment Considerations**
>
> Cholachgudda, K. E., Biradar, R. C., Kiran, B. M., & Prasannakumar, M. K. (2026). *Machine Learning with Applications*.

## 📖 Overview

This repository contains the code, experiment scripts, and trained model definitions for automated five-class severity classification (S0–S4) of grapevine downy mildew (*Plasmopara viticola*) from field-captured leaf images. The study systematically benchmarks **16 classification models**—5 classical ML pipelines and 11 end-to-end deep learning architectures—at 384 × 384 px input resolution under rigorous five-fold stratified cross-validation.

### Key Results (5-Fold CV, 384 × 384 px)

| Model | Accuracy (%) | κ_qw | F1 (%) | Model Size |
|-------|:------------:|:----:|:------:|:----------:|
| **ViT-S/16** | **87.81 ± 3.08** | **0.947** | **87.68** | 83.3 MB |
| MaxViT-T | 87.27 ± 1.87 | 0.951 | 87.33 | 116.6 MB |
| ConvNeXt-Tiny | 86.18 ± 1.33 | 0.946 | 86.04 | 106.2 MB |
| DINOv2 + LoRA | 85.96 ± 3.05 | 0.950 | 85.90 | 83.9 MB |
| FastViT-T8 | 85.96 ± 3.85 | 0.941 | 85.86 | 12.7 MB |
| EfficientNet-B0 | 85.85 ± 1.76 | 0.944 | 85.73 | 15.6 MB |
| **MobileNetV2** | **85.52 ± 3.09** | **0.937** | **85.44** | **8.7 MB** |
| EfficientNetV2-S | 85.09 ± 2.06 | 0.938 | 84.86 | 77.9 MB |
| ResNet-50 | 84.87 ± 2.88 | 0.935 | 84.67 | 90.0 MB |
| SVM-Linear (classical) | 70.18 ± 3.41 | 0.822 | 69.67 | — |

### Edge Deployment Results

| Platform | Model | Runtime | FPS | Cost |
|----------|-------|---------|:---:|:----:|
| Raspberry Pi 5 | MobileNetV2 | ONNX Runtime | **22.45** | USD 75 |
| Jetson Xavier NX | MobileNetV2 | PyTorch GPU | **48.8** | USD 399 |

### Highlights

- 🔬 **Field-captured dataset**: 920 images from 3 vineyard sites in southern Karnataka, India (split before augmentation to prevent data leakage) — [archived on Zenodo](https://doi.org/10.5281/zenodo.20701850)
- 🧠 **16-model benchmark**: 5 classical ML + 11 deep learning architectures (CNNs, ViTs, DINOv2, LoRA)
- 📏 **5-fold stratified cross-validation** as the primary evaluation metric
- 🔍 **Explainability**: Grad-CAM heatmaps with quantitative IoU validation against pathologist annotations
- 📊 **Statistical validation**: McNemar's test for pairwise model comparisons
- 🧪 **Ablation studies**: Resolution (224 vs 384 px), CORN ordinal regression, transfer learning
- 📱 **Edge deployment**: Empirical benchmarks on Raspberry Pi 5 and NVIDIA Jetson Xavier NX
- 📈 **t-SNE visualisation**: Feature space analysis for top-performing models

## 📁 Repository Structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── train_5fold_cv.py           # Primary 5-fold CV training (11 DL + 2 classical models)
│   ├── ablation_studies.py         # Resolution (224 vs 384) + CORN ordinal ablation
│   ├── gradcam_analysis.py         # Grad-CAM heatmaps + IoU validation
│   ├── tsne_visualization.py       # t-SNE feature space visualization
│   ├── edge_benchmark_jetson.py    # Jetson Xavier NX benchmarking
│   └── edge_benchmark_rpi5.py      # Raspberry Pi 5 benchmarking
└── notebooks/
    └── grapevine_downy_mildew_classification.ipynb  # Legacy notebook (reference)
```

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/kartikenc/grapevine-downy-mildew-classification.git
cd grapevine-downy-mildew-classification
pip install -r requirements.txt
```

### 2. Dataset Access

The dataset is publicly archived on Zenodo:

🔗 **[Zenodo: 10.5281/zenodo.20701850](https://doi.org/10.5281/zenodo.20701850)** — 920 field-captured grapevine leaf images with severity grading labels

Organise the downloaded images as:
```
data/
├── Original/
│   ├── O_0/   # S0: Healthy (0%, n=99)
│   ├── O_1/   # S1: Mild (1–25%, n=409)
│   ├── O_2/   # S2: Moderate (26–50%, n=172)
│   ├── O_3/   # S3: Severe (51–75%, n=132)
│   └── O_4/   # S4: Very Severe (>75%, n=108)
```

### 3. Run Training (5-Fold CV)

```bash
python src/train_5fold_cv.py
```

This script trains all 11 deep learning architectures + 2 classical ML models under 5-fold stratified cross-validation with checkpoint/resume support.

### 4. Inference with Pre-trained Models

```python
import torch
import timm
from torchvision import transforms
from PIL import Image

# Load ViT-S/16 (best model)
model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=5, img_size=384)
model.load_state_dict(torch.load('model_vit_s_16.pt', map_location='cpu'))
model.eval()

# Predict
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open('your_leaf_image.jpg').convert('RGB')
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax(1).item()

classes = ['S0 (Healthy)', 'S1 (Mild)', 'S2 (Moderate)', 'S3 (Severe)', 'S4 (Very Severe)']
print(f"Predicted: {classes[pred_class]}")
```

## 📊 Severity Classes

| Class | Severity | Infection Area | n (Images) | Management Action |
|-------|----------|:--------------:|:----------:|-------------------|
| S0 | Healthy | 0% | 99 | No intervention needed |
| S1 | Mild | 1–25% | 409 | Preventive copper-based fungicide |
| S2 | Moderate | 26–50% | 172 | Curative systemic fungicide |
| S3 | Severe | 51–75% | 132 | Intensive systemic fungicide |
| S4 | Very Severe | >75% | 108 | Defoliation management |

## 🔬 Ablation Studies

| Experiment | Finding |
|-----------|---------|
| Resolution (224 → 384 px) | +0.0 to +1.2 pp accuracy (all *p* > 0.30, non-significant) |
| CORN ordinal head vs CrossEntropy | No improvement (*p* ≥ 0.614) |
| Transfer learning removal | −26.28 pp accuracy (ImageNet init critical) |

## 🔍 Explainability (Grad-CAM)

Grad-CAM analysis confirms that models correctly attend to disease lesion regions, with quantitative IoU validation against pathologist-annotated binary lesion masks across six architectures showing a general increasing trend from S1 to S4. FastViT-T8 achieved the highest overall IoU (0.308 ± 0.113), while MobileNetV2 demonstrated a clear S1-to-S4 progression (0.090 → 0.406).

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{cholachgudda2026grapevine_classification,
  title={Explainable Deep Learning for Field-Deployable Grapevine Downy Mildew 
         Severity Classification: A Comparative Analysis with Edge-Deployment Considerations},
  author={Cholachgudda, Kartik E. and Biradar, Rajashekhar C. and Kiran, B. M. and Prasannakumar, M. K.},
  journal={Machine Learning with Applications},
  year={2026},
  publisher={Elsevier}
}

@dataset{cholachgudda2026grapevine_dataset,
  title={Grapevine Downy Mildew Severity Dataset: 920 Field-Captured Images 
         with Severity Grading Labels},
  author={Cholachgudda, Kartik E. and Biradar, Rajashekhar C. and Kiran, B. M. and Prasannakumar, M. K.},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.20701850},
  url={https://doi.org/10.5281/zenodo.20701850}
}
```

## 👥 Authors

- **Kartik E. Cholachgudda** — School of ECE, REVA University, Bengaluru ([ORCID](https://orcid.org/0000-0002-9217-6884))
- **Rajashekhar C. Biradar** — School of ECE, REVA University, Bengaluru
- **Kiran B. M.** — AgriHawk Technologies Private Limited (Fyllo), Bengaluru ([ORCID](https://orcid.org/0000-0001-6425-2336))
- **M. K. Prasannakumar** — PathoGenOmics Lab, Dept. of Plant Pathology, UAS GKVK, Bangalore ([ORCID](https://orcid.org/0000-0002-5115-411X))

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Karnataka Science and Technology Promotion Society (KSTePS) / VGST (Grant: GRD 954)
- Grape growers of Bengaluru Rural and Chikkaballapur districts, Karnataka
- PathoGenOmics Lab, Dept. of Plant Pathology, UAS GKVK, Bangalore
