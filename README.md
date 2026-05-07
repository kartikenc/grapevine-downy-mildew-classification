# Grapevine Downy Mildew Severity Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Explainable Deep Learning for Field-Deployable Grapevine Downy Mildew Severity Classification: A Comparative Analysis with Edge-Computing Implications**

## 📖 Overview

This repository contains the code and trained models for automated five-class severity classification (S0–S4) of grapevine downy mildew (*Plasmopara viticola*) from field-captured leaf images. The study benchmarks **9 classification models** spanning classical ML and deep learning approaches under identical experimental conditions.

### Key Results

| Model | Accuracy | F1-Score | Cohen's κ | Model Size |
|-------|:--------:|:--------:|:---------:|:----------:|
| **EfficientNet-B0** | **92.57%** | **92.45%** | **0.907** | 15.6 MB |
| ResNet-50 | 90.86% | 90.77% | 0.886 | 90.0 MB |
| VGG-16 | 90.86% | 90.92% | 0.886 | 512.3 MB |
| MobileNetV2 | 89.14% | 89.01% | 0.864 | **8.7 MB** |
| Gradient Boosting | 81.14% | 81.47% | 0.764 | 3.5 MB |

### Features
- 🔬 **Field-captured dataset**: 920 images from Bagalkot vineyards (augmented to 1,750)
- 🧠 **9 model benchmark**: SVM, RF, GB, KNN + ResNet-50, VGG-16, MobileNetV2, EfficientNet-B0
- 🔍 **Explainability**: Grad-CAM heatmaps + t-SNE feature space visualisation
- 📊 **Statistical validation**: McNemar's test for pairwise comparisons
- 🧪 **Ablation study**: Transfer learning, augmentation, dropout, LR scheduling
- 📱 **Edge deployment**: Accuracy vs. model size analysis for portable devices

## 📁 Repository Structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Dataset statistics and visualization
│   ├── 02_classical_ml_training.ipynb   # Classical ML pipeline
│   ├── 03_deep_learning_training.ipynb  # Deep learning training pipeline
│   ├── 04_gradcam_analysis.ipynb        # Grad-CAM explainability analysis
│   └── 05_results_analysis.ipynb        # Results, McNemar's test, ablation
├── src/
│   ├── data_utils.py                    # Data loading and augmentation
│   ├── models.py                        # Model definitions
│   └── evaluation.py                    # Evaluation metrics and plotting
├── models/                              # Pre-trained model weights
│   ├── efficientnet_b0.pt
│   └── mobilenetv2.pt
└── results/                             # Experiment results and figures
    ├── figures/
    └── metrics/
```

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/kartikenc/grapevine-downy-mildew-classification.git
cd grapevine-downy-mildew-classification
pip install -r requirements.txt
```

### 2. Dataset Access

The dataset is available upon request. Please contact the corresponding author at **kartikenc@gmail.com** or request access via Kaggle:

🔗 [Kaggle Dataset](https://www.kaggle.com/datasets/kartikec/grapevine-downy-mildew-severity)

The dataset should be organized as:
```
data/
├── train/
│   ├── S0/   # Healthy (0% infection)
│   ├── S1/   # Slight (1-10%)
│   ├── S2/   # Moderate (11-25%)
│   ├── S3/   # Severe (26-50%)
│   └── S4/   # Very Severe (>50%)
├── val/
└── test/
```

### 3. Run Notebooks

Open and run the Jupyter notebooks in order:

```bash
jupyter notebook notebooks/
```

### 4. Inference with Pre-trained Models

```python
import torch
from torchvision import transforms, models
from PIL import Image

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier[-1] = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(1280, 5)
)
model.load_state_dict(torch.load('models/efficientnet_b0.pt', map_location='cpu'))
model.eval()

# Predict
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open('your_leaf_image.jpg').convert('RGB')
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax(1).item()

classes = ['S0 (Healthy)', 'S1 (Slight)', 'S2 (Moderate)', 'S3 (Severe)', 'S4 (Very Severe)']
print(f"Predicted: {classes[pred_class]}")
```

## 📊 Severity Classes

| Class | Severity | Infection (%) | Management Action |
|-------|----------|:------------:|-------------------|
| S0 | Healthy | 0% | No intervention needed |
| S1 | Slight | 1–10% | Preventive copper-based fungicide |
| S2 | Moderate | 11–25% | Curative fungicide application |
| S3 | Severe | 26–50% | Intensive systemic fungicide |
| S4 | Very Severe | >50% | Defoliation management |

## 🔍 Grad-CAM Explainability

The Grad-CAM analysis confirms that EfficientNet-B0 correctly focuses on disease lesion regions across all severity classes, providing interpretable visual evidence for agricultural deployment.

![Grad-CAM Visualization](results/figures/fig_gradcam_efficientnet_b0.png)

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{cholachgudda2026grapevine,
  title={Explainable Deep Learning for Field-Deployable Grapevine Downy Mildew 
         Severity Classification: A Comparative Analysis with Edge-Computing Implications},
  author={Cholachgudda, Kartik E. and Biradar, Rajashekhar C. and Kiran, B. M. and Prasannakumar, M. K.},
  journal={Artificial Intelligence in Agriculture},
  year={2026},
  publisher={KeAi/Elsevier}
}
```

## 👥 Authors

- **Kartik E. Cholachgudda** — School of ECE, REVA University, Bengaluru
- **Rajashekhar C. Biradar** — School of ECE, REVA University, Bengaluru
- **Kiran B. M.** — PathoGenOmics Lab, Dept. of Plant Pathology, UAS GKVK, Bangalore
- **M. K. Prasannakumar** — PathoGenOmics Lab, Dept. of Plant Pathology, UAS GKVK, Bangalore

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Karnataka Science and Technology Promotion Society (KSTePS) / VGST (Grant: GRD 954)
- Grape growers of Bagalkot district, Karnataka
- PathoGenOmics Lab, Dept. of Plant Pathology, UAS GKVK, Bangalore
