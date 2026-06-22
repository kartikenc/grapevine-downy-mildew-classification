#!/usr/bin/env python3
"""
10_gradcam_iou_analysis.py
===========================
Phase 2: Quantitative XAI — Grad-CAM IoU Analysis

Computes the Intersection-over-Union (IoU) between:
  - Pathologist-annotated binary lesion masks (from LabelMe JSON polygons)
  - Grad-CAM activation maps (thresholded) from EfficientNet-B0 and MobileNetV2

Produces:
  - Per-image IoU scores
  - Per-class mean IoU
  - Overall mean IoU with 95% CI
  - Publication-ready figure: side-by-side original / mask / Grad-CAM / overlay
  - JSON results for manuscript integration

Author: Kartik E. Cholachgudda
Date: May 2026
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
import cv2

# Try importing grad-cam library
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False
    print("FATAL: pytorch-grad-cam not installed. Install with: pip install grad-cam")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================
DATASET_ROOT = Path(r"D:\Projects\AgRECA\PhD\PhD2\04_Dataset\Balanced_Dataset_v2\splits")
GRAD_DIR = Path(r"D:\Projects\AgRECA\PhD\PhD2\04_Dataset\Balanced_Dataset\splits\grad")  # LabelMe annotations
DL_RESULTS_DIR = Path(r"D:\Projects\AgRECA\PhD\PhD2\03_Experiments\results\paper2_clean_split_v2")
OUTPUT_DIR = Path(r"D:\Projects\AgRECA\PhD\PhD2\02_Publications\Journal_Papers_InProgress\P02_Downy_Mildew_AI_Agriculture\figures")

CLASSES = ['S1', 'S2', 'S3', 'S4']  # S0 excluded (no lesions)
CLASS_LABELS = {
    'S1': 'S1 (Slight, 1-25%)',
    'S2': 'S2 (Moderate, 26-50%)',
    'S3': 'S3 (Severe, 51-75%)',
    'S4': 'S4 (Very Severe, 76-100%)',
}
NUM_CLASSES = 5  # Model still has 5 outputs (S0-S4)
IMAGE_SIZE = 384
DINOV2_SIZE = 392
GRADCAM_THRESHOLD = 0.5  # Binarisation threshold for Grad-CAM heatmaps
RANDOM_SEED = 42

# Top 5 models for Grad-CAM analysis
GRADCAM_MODELS = [
    'ViT-S/16', 'MaxViT-T', 'ConvNeXt-Tiny', 'FastViT-T8', 'EfficientNet-B0',
]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})


# ============================================================
# Model Loading
# ============================================================

def get_model(name):
    """Get a model with modified classifier for NUM_CLASSES."""
    import timm
    if name == 'EfficientNet-B0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[-1].in_features, NUM_CLASSES))
    elif name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[-1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[-1].in_features, NUM_CLASSES))
    elif name == 'ResNet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, NUM_CLASSES))
    elif name == 'ViT-S/16':
        model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=NUM_CLASSES, img_size=IMAGE_SIZE)
    elif name == 'ConvNeXt-Tiny':
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[2].in_features, NUM_CLASSES))
    elif name == 'EfficientNetV2-S':
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[-1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[-1].in_features, NUM_CLASSES))
    elif name == 'MaxViT-T':
        model = timm.create_model('maxvit_tiny_tf_384', pretrained=False, num_classes=NUM_CLASSES)
    elif name == 'FastViT-T8':
        model = timm.create_model('fastvit_t8', pretrained=False, num_classes=NUM_CLASSES)
    elif name == 'MobileNetV4-S':
        model = timm.create_model('mobilenetv4_conv_small', pretrained=False, num_classes=NUM_CLASSES)
    elif name == 'DINOv2-ViTS14':
        model = timm.create_model('vit_small_patch14_dinov2', pretrained=False, num_classes=NUM_CLASSES, img_size=DINOV2_SIZE)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def get_model_image_size(name):
    if 'DINOv2' in name:
        return DINOV2_SIZE
    return IMAGE_SIZE


def load_trained_model(name):
    """Load a trained model from disk."""
    model = get_model(name)
    safe_name = name.replace('-', '_').replace('/', '_').lower()
    weight_path = DL_RESULTS_DIR / f'model_{safe_name}.pt'
    if not weight_path.exists():
        print(f"  ERROR: Model weights not found: {weight_path}")
        return None
    state = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    # Handle potential key mismatches gracefully
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
        print(f"  [WARN] Loaded {name} with strict=False")
    model.to(DEVICE)
    model.eval()
    return model


def get_target_layers(model, name):
    """Get the target layer for Grad-CAM based on model architecture."""
    if name == 'EfficientNet-B0':
        return [model.features[-1]]
    elif name == 'MobileNetV2':
        return [model.features[-1]]
    elif name == 'ResNet50':
        return [model.layer4[-1]]
    elif name == 'ViT-S/16':
        return [model.blocks[-1].norm1]
    elif name == 'ConvNeXt-Tiny':
        return [model.features[-1][-1]]
    elif name == 'EfficientNetV2-S':
        return [model.features[-1]]
    elif name == 'MaxViT-T':
        return [model.stages[-1].blocks[-1]]
    elif name == 'FastViT-T8':
        return [model.stages[-1].blocks[-1]]
    elif name == 'MobileNetV4-S':
        return [model.blocks[-1]]
    elif name == 'DINOv2-ViTS14':
        return [model.blocks[-1].norm1]
    raise ValueError(f"Unknown model for Grad-CAM target layers: {name}")


# ============================================================
# LabelMe → Binary Mask Conversion
# ============================================================

def labelme_json_to_mask(json_path, target_size=IMAGE_SIZE):
    """
    Convert LabelMe JSON annotations to a binary mask.

    Args:
        json_path: Path to the LabelMe JSON file
        target_size: Output mask size (square)

    Returns:
        binary_mask: numpy array (target_size x target_size), 0 or 1
        original_size: (width, height) of original image
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = data['imageWidth']
    img_h = data['imageHeight']

    # Create mask at original resolution
    mask = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)

    for shape in data.get('shapes', []):
        if shape['label'].lower() == 'lesion' and shape['shape_type'] == 'polygon':
            points = [(p[0], p[1]) for p in shape['points']]
            if len(points) >= 3:
                draw.polygon(points, fill=255)

    # Resize to target size using nearest-neighbour (preserve binary nature)
    mask_resized = mask.resize((target_size, target_size), Image.NEAREST)
    binary_mask = (np.array(mask_resized) > 127).astype(np.uint8)

    return binary_mask, (img_w, img_h)


# ============================================================
# Grad-CAM Generation
# ============================================================

def generate_gradcam_mask(model, target_layers, img_path, class_idx, model_name='', threshold=GRADCAM_THRESHOLD):
    """
    Generate a binary Grad-CAM mask for a single image.

    Args:
        model: PyTorch model
        target_layers: List of target layers for Grad-CAM
        img_path: Path to the input image
        class_idx: Target class index (0-4 for S0-S4)
        threshold: Binarisation threshold

    Returns:
        binary_cam: numpy array (sz x sz), 0 or 1
        grayscale_cam: numpy array (sz x sz), float 0-1
        rgb_img: numpy array (sz x sz x 3), float 0-1
    """
    import math
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    sz = get_model_image_size(model_name) if model_name else IMAGE_SIZE

    eval_transform = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    pil_img = Image.open(img_path).convert('RGB')
    pil_img_resized = pil_img.resize((sz, sz))
    rgb_img = np.array(pil_img_resized) / 255.0

    input_tensor = eval_transform(pil_img).unsqueeze(0).to(DEVICE)

    # ViT/transformer models need reshape_transform for Grad-CAM
    def vit_reshape_transform(tensor, height=None, width=None):
        """Reshape (B, N+1, C) -> (B, C, H, W), dropping CLS token."""
        if tensor.dim() == 3:
            # Remove CLS token (first token)
            result = tensor[:, 1:, :]
            n_tokens = result.shape[1]
            h = w = int(math.sqrt(n_tokens))
            result = result.reshape(result.shape[0], h, w, result.shape[2])
            result = result.permute(0, 3, 1, 2)  # (B, C, H, W)
            return result
        return tensor

    is_vit = model_name in ('ViT-S/16', 'DINOv2-ViTS14', 'DINOv2-ViTS14-LoRA')
    reshape_fn = vit_reshape_transform if is_vit else None

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_fn)
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Binarise
    binary_cam = (grayscale_cam >= threshold).astype(np.uint8)

    return binary_cam, grayscale_cam, rgb_img


# ============================================================
# IoU Computation
# ============================================================

def compute_iou(mask_gt, mask_pred):
    """
    Compute Intersection-over-Union between two binary masks.

    Args:
        mask_gt: Ground truth binary mask (numpy array)
        mask_pred: Predicted binary mask (numpy array)

    Returns:
        iou: float, IoU score (0-1)
        intersection: int, number of intersection pixels
        union: int, number of union pixels
    """
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    union = np.logical_or(mask_gt, mask_pred).sum()

    if union == 0:
        return 0.0, 0, 0

    iou = intersection / union
    return float(iou), int(intersection), int(union)


def compute_dice(mask_gt, mask_pred):
    """Compute Dice coefficient (F1 for segmentation)."""
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    total = mask_gt.sum() + mask_pred.sum()

    if total == 0:
        return 0.0

    return float(2 * intersection / total)


# ============================================================
# Main Pipeline
# ============================================================

def run_gradcam_iou(model_name='EfficientNet-B0'):
    """
    Run the full Grad-CAM IoU pipeline for a given model.

    Returns:
        results: dict with per-image and aggregate metrics
    """
    print(f"\n{'='*60}")
    print(f"  Grad-CAM IoU Analysis: {model_name}")
    print(f"{'='*60}")

    # Load model
    model = load_trained_model(model_name)
    if model is None:
        return None
    target_layers = get_target_layers(model, model_name)

    # Class index mapping (S0=0, S1=1, ..., S4=4)
    class_to_idx = {'S0': 0, 'S1': 1, 'S2': 2, 'S3': 3, 'S4': 4}

    all_results = []
    class_results = {cls: [] for cls in CLASSES}

    for cls in CLASSES:
        cls_dir = GRAD_DIR / cls
        json_files = sorted(cls_dir.glob('*.json'))
        class_idx = class_to_idx[cls]

        print(f"\n  Processing {cls} ({len(json_files)} images)...")

        for jf in json_files:
            # Find corresponding image
            json_data = json.load(open(jf, 'r', encoding='utf-8'))
            img_name = json_data.get('imagePath', '')
            img_path = cls_dir / img_name

            # Handle case sensitivity
            if not img_path.exists():
                # Try common extensions
                for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
                    candidate = cls_dir / (jf.stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break

            if not img_path.exists():
                print(f"    WARNING: Image not found for {jf.name}, skipping")
                continue

            # 1. Convert LabelMe annotation to binary mask
            gt_mask, orig_size = labelme_json_to_mask(jf, IMAGE_SIZE)

            # 2. Generate Grad-CAM mask
            cam_mask, cam_raw, rgb_img = generate_gradcam_mask(
                model, target_layers, img_path, class_idx, model_name=model_name
            )

            # 3. Compute metrics
            iou, inter, union = compute_iou(gt_mask, cam_mask)
            dice = compute_dice(gt_mask, cam_mask)

            # Lesion coverage stats
            gt_coverage = gt_mask.sum() / gt_mask.size * 100
            cam_coverage = cam_mask.sum() / cam_mask.size * 100

            result = {
                'image': img_path.name,
                'class': cls,
                'iou': round(iou, 4),
                'dice': round(dice, 4),
                'intersection_px': inter,
                'union_px': union,
                'gt_coverage_pct': round(gt_coverage, 2),
                'cam_coverage_pct': round(cam_coverage, 2),
                'orig_size': list(orig_size),
            }

            all_results.append(result)
            class_results[cls].append(result)

            print(f"    {img_path.name}: IoU={iou:.3f}, Dice={dice:.3f}, "
                  f"GT={gt_coverage:.1f}%, CAM={cam_coverage:.1f}%")

    # ── Aggregate Metrics ──
    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS — {model_name}")
    print(f"{'='*60}")

    aggregate = {}
    all_ious = [r['iou'] for r in all_results]
    all_dices = [r['dice'] for r in all_results]

    # Overall
    mean_iou = np.mean(all_ious)
    std_iou = np.std(all_ious, ddof=1)
    # Bootstrap 95% CI
    bootstrap_ious = []
    for _ in range(10000):
        sample = np.random.choice(all_ious, size=len(all_ious), replace=True)
        bootstrap_ious.append(np.mean(sample))
    ci_low = np.percentile(bootstrap_ious, 2.5)
    ci_high = np.percentile(bootstrap_ious, 97.5)

    aggregate['overall'] = {
        'mean_iou': round(float(mean_iou), 4),
        'std_iou': round(float(std_iou), 4),
        'ci_95_low': round(float(ci_low), 4),
        'ci_95_high': round(float(ci_high), 4),
        'mean_dice': round(float(np.mean(all_dices)), 4),
        'std_dice': round(float(np.std(all_dices, ddof=1)), 4),
        'n_images': len(all_results),
    }
    print(f"\n  Overall: IoU = {mean_iou:.4f} +/- {std_iou:.4f} "
          f"(95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"           Dice = {np.mean(all_dices):.4f} +/- {np.std(all_dices, ddof=1):.4f}")

    # Per-class
    print(f"\n  {'Class':<12} {'n':>3} {'Mean IoU':>10} {'Std':>8} {'Mean Dice':>10}")
    print(f"  {'-'*48}")
    for cls in CLASSES:
        cls_ious = [r['iou'] for r in class_results[cls]]
        cls_dices = [r['dice'] for r in class_results[cls]]
        if cls_ious:
            m = np.mean(cls_ious)
            s = np.std(cls_ious, ddof=1) if len(cls_ious) > 1 else 0.0
            md = np.mean(cls_dices)
            aggregate[cls] = {
                'mean_iou': round(float(m), 4),
                'std_iou': round(float(s), 4),
                'mean_dice': round(float(md), 4),
                'n_images': len(cls_ious),
            }
            print(f"  {CLASS_LABELS[cls]:<30} {len(cls_ious):>3} {m:>10.4f} {s:>8.4f} {md:>10.4f}")

    return {
        'model': model_name,
        'threshold': GRADCAM_THRESHOLD,
        'timestamp': datetime.now().isoformat(),
        'per_image': all_results,
        'aggregate': aggregate,
    }


# ============================================================
# Visualisation
# ============================================================

def plot_iou_examples(results_eb0, results_mv2, output_dir):
    """
    Create a publication figure showing 4 examples (one per class)
    with original image, GT mask, Grad-CAM mask, and overlay.
    """
    print("\n  Generating Grad-CAM IoU visualisation figure...")

    # Pick 1 representative image per class (median IoU)
    examples = {}
    for cls in CLASSES:
        cls_results = [r for r in results_eb0['per_image'] if r['class'] == cls]
        cls_results.sort(key=lambda x: x['iou'])
        # Pick median
        mid = len(cls_results) // 2
        examples[cls] = cls_results[mid]

    # Load models
    model_eb0 = load_trained_model('EfficientNet-B0')
    tl_eb0 = get_target_layers(model_eb0, 'EfficientNet-B0')

    model_mv2 = load_trained_model('MobileNetV2')
    tl_mv2 = get_target_layers(model_mv2, 'MobileNetV2')

    class_to_idx = {'S1': 1, 'S2': 2, 'S3': 3, 'S4': 4}

    fig, axes = plt.subplots(4, 5, figsize=(18, 14))
    col_titles = ['Original Image', 'Ground Truth\nLesion Mask',
                  'EfficientNet-B0\nGrad-CAM', 'MobileNetV2\nGrad-CAM',
                  'EB0 Overlay\n(GT=green, CAM=red)']

    for row, cls in enumerate(CLASSES):
        ex = examples[cls]
        img_path = GRAD_DIR / cls / ex['image']
        json_path = GRAD_DIR / cls / (Path(ex['image']).stem + '.json')

        # Handle case sensitivity for image path
        if not img_path.exists():
            for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
                candidate = GRAD_DIR / cls / (Path(ex['image']).stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

        class_idx = class_to_idx[cls]

        # Ground truth mask
        gt_mask, _ = labelme_json_to_mask(json_path, IMAGE_SIZE)

        # Grad-CAM masks
        cam_eb0, cam_raw_eb0, rgb_img = generate_gradcam_mask(
            model_eb0, tl_eb0, img_path, class_idx)
        cam_mv2, cam_raw_mv2, _ = generate_gradcam_mask(
            model_mv2, tl_mv2, img_path, class_idx)

        # IoU for labels
        iou_eb0 = compute_iou(gt_mask, cam_eb0)[0]
        iou_mv2 = compute_iou(gt_mask, cam_mv2)[0]

        # Col 0: Original
        axes[row, 0].imshow(rgb_img)
        axes[row, 0].set_ylabel(CLASS_LABELS[cls], fontsize=11, fontweight='bold',
                                rotation=0, labelpad=90, va='center')

        # Col 1: Ground truth mask
        axes[row, 1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)

        # Col 2: EfficientNet-B0 Grad-CAM (raw heatmap)
        eb0_overlay = show_cam_on_image(rgb_img.astype(np.float32), cam_raw_eb0, use_rgb=True)
        axes[row, 2].imshow(eb0_overlay)
        axes[row, 2].set_xlabel(f'IoU = {iou_eb0:.3f}', fontsize=10, fontweight='bold')

        # Col 3: MobileNetV2 Grad-CAM (raw heatmap)
        mv2_overlay = show_cam_on_image(rgb_img.astype(np.float32), cam_raw_mv2, use_rgb=True)
        axes[row, 3].imshow(mv2_overlay)
        axes[row, 3].set_xlabel(f'IoU = {iou_mv2:.3f}', fontsize=10, fontweight='bold')

        # Col 4: Overlap visualisation (GT=green, CAM=red, overlap=yellow)
        overlap_img = (rgb_img * 0.4).copy()  # Dim background
        # Green channel for GT
        gt_overlay = np.zeros_like(rgb_img)
        gt_overlay[:, :, 1] = gt_mask * 0.6  # Green for GT
        # Red channel for Grad-CAM
        cam_overlay = np.zeros_like(rgb_img)
        cam_overlay[:, :, 0] = cam_eb0 * 0.6  # Red for CAM
        combined = np.clip(overlap_img + gt_overlay + cam_overlay, 0, 1)
        axes[row, 4].imshow(combined)

        for col in range(5):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    # Column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=10)

    # Legend for overlap column
    legend_patches = [
        mpatches.Patch(color='green', alpha=0.6, label='Ground Truth (Pathologist)'),
        mpatches.Patch(color='red', alpha=0.6, label='Grad-CAM Activation'),
        mpatches.Patch(color='yellow', alpha=0.6, label='Overlap (Intersection)'),
    ]
    axes[3, 4].legend(handles=legend_patches, loc='lower center',
                       fontsize=8, framealpha=0.8)

    plt.suptitle('Grad-CAM Localisation Fidelity: Ground Truth vs. Model Activation',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.07, 0, 1, 0.96])

    plt.savefig(output_dir / 'fig_gradcam_iou_validation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_gradcam_iou_validation.tiff', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Grad-CAM IoU figure saved")


def plot_iou_bar_chart(results_eb0, results_mv2, output_dir):
    """Bar chart comparing per-class IoU for both models."""
    print("  Generating IoU bar chart...")

    classes = CLASSES
    eb0_means = [results_eb0['aggregate'][c]['mean_iou'] for c in classes]
    eb0_stds = [results_eb0['aggregate'][c]['std_iou'] for c in classes]
    mv2_means = [results_mv2['aggregate'][c]['mean_iou'] for c in classes]
    mv2_stds = [results_mv2['aggregate'][c]['std_iou'] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, eb0_means, width, yerr=eb0_stds,
                   label='EfficientNet-B0', color='#2196F3', capsize=5, alpha=0.85)
    bars2 = ax.bar(x + width/2, mv2_means, width, yerr=mv2_stds,
                   label='MobileNetV2', color='#FF9800', capsize=5, alpha=0.85)

    # Overall means as horizontal lines
    eb0_overall = results_eb0['aggregate']['overall']['mean_iou']
    mv2_overall = results_mv2['aggregate']['overall']['mean_iou']
    ax.axhline(y=eb0_overall, color='#2196F3', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'EB0 Overall Mean ({eb0_overall:.3f})')
    ax.axhline(y=mv2_overall, color='#FF9800', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'MV2 Overall Mean ({mv2_overall:.3f})')

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Severity Class', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean IoU', fontweight='bold', fontsize=12)
    ax.set_title('Grad-CAM Localisation Fidelity (IoU) by Severity Class', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in classes])
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_gradcam_iou_barchart.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_gradcam_iou_barchart.tiff', dpi=300, bbox_inches='tight')
    plt.close()
    print("  IoU bar chart saved")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 2: QUANTITATIVE XAI — GRAD-CAM IoU ANALYSIS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Annotation dir: {GRAD_DIR}")
    print(f"Threshold: {GRADCAM_THRESHOLD}")
    print(f"Models: {', '.join(GRADCAM_MODELS)}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run analysis for all models ──
    all_results = {}
    for model_name in GRADCAM_MODELS:
        result = run_gradcam_iou(model_name)
        if result is not None:
            all_results[model_name] = result
        else:
            print(f"  [SKIP] {model_name}: model loading failed")

    if len(all_results) == 0:
        print("ERROR: No models loaded successfully. Aborting.")
        return

    # ── Save JSON results ──
    results_path = OUTPUT_DIR / 'gradcam_iou_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {results_path}")

    # ── Generate multi-model bar chart ──
    print("\n  Generating multi-model IoU bar chart...")
    model_names = list(all_results.keys())
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(max(10, 3*n_models), 6))
    x = np.arange(len(CLASSES))
    width = 0.8 / n_models
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#E91E63',
              '#00BCD4', '#FF5722', '#3F51B5', '#8BC34A', '#FFC107']

    for i, mname in enumerate(model_names):
        means = [all_results[mname]['aggregate'].get(c, {}).get('mean_iou', 0) for c in CLASSES]
        stds = [all_results[mname]['aggregate'].get(c, {}).get('std_iou', 0) for c in CLASSES]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      label=mname, color=colors[i % len(colors)], capsize=3, alpha=0.85)

    ax.set_xlabel('Severity Class', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean IoU', fontweight='bold', fontsize=12)
    ax.set_title('Grad-CAM Localisation Fidelity (IoU) by Severity Class', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in CLASSES])
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_gradcam_iou_barchart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  IoU bar chart saved")

    # ── Print summary table ──
    print("\n" + "=" * 80)
    print("GRAD-CAM IoU SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Overall IoU':<16} {'95% CI':<20} {'N images'}")
    print("-" * 80)
    for mname in model_names:
        agg = all_results[mname]['aggregate']['overall']
        print(f"{mname:<20} {agg['mean_iou']:.3f} ± {agg['std_iou']:.3f}    "
              f"[{agg['ci_95_low']:.3f}, {agg['ci_95_high']:.3f}]    {agg['n_images']}")

    print("\n  ALL DONE.")


if __name__ == '__main__':
    main()

