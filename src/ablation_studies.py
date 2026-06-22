#!/usr/bin/env python3
"""
14_ablation_studies.py
======================
Two ablation studies for the AIIA journal revision:

  Ablation A — Resolution: 224px vs 384px (4 overlapping models)
  Ablation B — CORN Ordinal Head vs CrossEntropy on ViT-S/16 at 384px

Author: Kartik E. Cholachgudda
Date:   June 2026 (AIIA review revision)
"""

import os
import sys
import time
import json
import copy
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

# ============================================================
# Logging — tee to console + file
# ============================================================
LOG_FILE = Path(r"d:\Projects\AgRECA\PhD\PhD2\03_Experiments\results\paper2_clean_split_v2\ablation_log.txt")

class LogTee:
    """Write to both console and file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.__stdout__
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(filepath, 'a', encoding='utf-8')
    def write(self, msg):
        try:
            self.terminal.write(msg)
            self.terminal.flush()
        except:
            pass
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        try: self.terminal.flush()
        except: pass
        self.log.flush()

sys.stdout = LogTee(LOG_FILE)
sys.stderr = LogTee(LOG_FILE)

if sys.platform == 'win32':
    pass  # LogTee handles encoding

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error
from scipy import stats
from PIL import Image
import albumentations as A
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Configuration (matches 11_clean_5fold_cv.py)
# ============================================================
ORIGINALS_ROOT = Path(r"d:\Projects\AgRECA\PhD\PhD2\04_Dataset\Downy_Mildew\Original")
RESULTS_DIR    = Path(r"d:\Projects\AgRECA\PhD\PhD2\03_Experiments\results\paper2_clean_split_v2")
FIGURES_DIR    = Path(r"d:\Projects\AgRECA\PhD\PhD2\02_Publications\Journal_Papers_InProgress\P02_Downy_Mildew_AI_Agriculture\figures")

CLASS_MAP   = {'O_0': 0, 'O_1': 1, 'O_2': 2, 'O_3': 3, 'O_4': 4}
CLASSES     = ['S0', 'S1', 'S2', 'S3', 'S4']
NUM_CLASSES = 5

IMAGE_SIZE    = 384
BATCH_SIZE    = 16
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
RANDOM_SEED   = 42
N_FOLDS       = 5

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}

# Checkpoint paths for existing results
CKPT_224 = RESULTS_DIR / 'cv_5fold_checkpoint_224px_backup.json'
CKPT_384 = RESULTS_DIR / 'cv_5fold_checkpoint.json'

# Output paths
ABLATION_RES_JSON  = RESULTS_DIR / 'ablation_resolution.json'
ABLATION_CORN_JSON = RESULTS_DIR / 'ablation_corn.json'
ABLATION_CORN_CKPT = RESULTS_DIR / 'ablation_corn_checkpoint.json'
ABLATION_RES_FIG   = FIGURES_DIR / 'ablation_resolution.png'

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# Augmentation pipeline (identical to 11_clean_5fold_cv.py)
# ============================================================
aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    ], p=0.6),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.GaussNoise(p=1.0),
    ], p=0.2),
    A.Affine(translate_percent=(-0.06, 0.06), scale=(0.9, 1.1), rotate=(-15, 15), p=0.3),
])


# ============================================================
# Dataset class (identical to 11_clean_5fold_cv.py)
# ============================================================
class OriginalImageDataset(Dataset):
    """Dataset that loads original images with optional augmentation."""
    def __init__(self, image_paths, labels, transform=None, augment_to_balance=False, target_per_class=None):
        self.transform = transform

        if augment_to_balance and target_per_class:
            self.image_paths = list(image_paths)
            self.labels = list(labels)
            self.is_augmented = [False] * len(image_paths)

            class_counts = Counter(labels)
            for cls, count in class_counts.items():
                if count < target_per_class:
                    cls_indices = [i for i, l in enumerate(labels) if l == cls]
                    needed = target_per_class - count
                    rng = random.Random(RANDOM_SEED + cls)
                    for _ in range(needed):
                        src_idx = rng.choice(cls_indices)
                        self.image_paths.append(image_paths[src_idx])
                        self.labels.append(cls)
                        self.is_augmented.append(True)
        else:
            self.image_paths = list(image_paths)
            self.labels = list(labels)
            self.is_augmented = [False] * len(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')

        if self.is_augmented[idx]:
            img_np = np.array(img)
            img_np = aug_pipeline(image=img_np)['image']
            img = Image.fromarray(img_np)

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


# ============================================================
# Shared helpers
# ============================================================
def get_transforms(img_size=None):
    """Get train/eval transforms for a given resolution."""
    sz = img_size or IMAGE_SIZE
    train_transform = transforms.Compose([
        transforms.Resize((sz + 32, sz + 32)),
        transforms.RandomCrop(sz),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def load_originals():
    """Load all 920 original images (paths + labels)."""
    paths, labels = [], []
    for dir_name, cls_idx in CLASS_MAP.items():
        src_dir = ORIGINALS_ROOT / dir_name
        for f in sorted(src_dir.iterdir()):
            if f.is_file() and f.suffix in IMAGE_EXTENSIONS:
                try:
                    img = Image.open(f)
                    img.verify()
                    paths.append(str(f))
                    labels.append(cls_idx)
                except Exception:
                    print(f"  Skipping corrupt: {f.name}")
    return paths, np.array(labels)


def convert(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj


def save_json(data, path):
    """Save dict to JSON with numpy conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(json.loads(json.dumps(data, default=convert)), f, indent=2)
    print(f"  Saved: {path.name}")


def setup_mpl():
    """Publication-quality matplotlib settings."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ╔════════════════════════════════════════════════════════════╗
# ║  ABLATION A — Resolution (224px vs 384px)                 ║
# ╚════════════════════════════════════════════════════════════╝
def ablation_resolution():
    """Compare 224px vs 384px for overlapping models."""
    print("\n" + "=" * 70)
    print("  ABLATION A — Resolution: 224px vs 384px")
    print("=" * 70)

    # Load results
    with open(CKPT_224, 'r') as f:
        data_224 = json.load(f)
    with open(CKPT_384, 'r') as f:
        data_384 = json.load(f)

    # Overlapping models
    overlap_models = ['ResNet50', 'MobileNetV2', 'EfficientNet-B0', 'ViT-S/16']

    # Verify both files contain these models
    for m in overlap_models:
        assert m in data_224, f"Model {m} not found in 224px results"
        assert m in data_384, f"Model {m} not found in 384px results"

    # Build comparison table
    table = {}
    for model in overlap_models:
        folds_224 = data_224[model]
        folds_384 = data_384[model]

        accs_224 = np.array([f['accuracy'] for f in folds_224])
        accs_384 = np.array([f['accuracy'] for f in folds_384])
        kappas_224 = np.array([f['quadratic_weighted_kappa'] for f in folds_224])
        kappas_384 = np.array([f['quadratic_weighted_kappa'] for f in folds_384])

        # Paired t-test on per-fold accuracies
        t_stat, p_val = stats.ttest_rel(accs_384, accs_224)

        delta_acc = np.mean(accs_384) - np.mean(accs_224)

        table[model] = {
            '224px': {
                'mean_acc': float(np.mean(accs_224)),
                'std_acc': float(np.std(accs_224)),
                'mean_kappa_qw': float(np.mean(kappas_224)),
                'std_kappa_qw': float(np.std(kappas_224)),
                'per_fold_acc': accs_224.tolist(),
                'per_fold_kappa_qw': kappas_224.tolist(),
            },
            '384px': {
                'mean_acc': float(np.mean(accs_384)),
                'std_acc': float(np.std(accs_384)),
                'mean_kappa_qw': float(np.mean(kappas_384)),
                'std_kappa_qw': float(np.std(kappas_384)),
                'per_fold_acc': accs_384.tolist(),
                'per_fold_kappa_qw': kappas_384.tolist(),
            },
            'delta_acc': float(delta_acc),
            'delta_acc_pct': float(delta_acc * 100),
            'paired_ttest': {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant_005': bool(p_val < 0.05),
            },
        }

    # Print table
    print(f"\n  {'Model':<18} {'224px Acc':>12} {'384px Acc':>12} {'Δ Acc':>10} {'κ_qw 224':>10} {'κ_qw 384':>10} {'p-value':>10} {'Sig.':>5}")
    print("  " + "-" * 95)
    for model, r in table.items():
        d224 = r['224px']
        d384 = r['384px']
        p = r['paired_ttest']
        sig = "*" if p['significant_005'] else "ns"
        print(f"  {model:<18} {d224['mean_acc']*100:>5.2f}±{d224['std_acc']*100:.2f}%  "
              f"{d384['mean_acc']*100:>5.2f}±{d384['std_acc']*100:.2f}%  "
              f"{r['delta_acc_pct']:>+6.2f}pp   "
              f"{d224['mean_kappa_qw']:.4f}    {d384['mean_kappa_qw']:.4f}    "
              f"{p['p_value']:.4f}  {sig:>4}")

    # Save results
    output = {
        'description': 'Ablation A — Resolution comparison (224px vs 384px)',
        'timestamp': datetime.now().isoformat(),
        'overlapping_models': overlap_models,
        'results': table,
    }
    save_json(output, ABLATION_RES_JSON)

    # ── Generate grouped bar chart ──
    setup_mpl()

    models_short = ['ResNet50', 'MobileNetV2', 'EffNet-B0', 'ViT-S/16']
    accs_224_list = [table[m]['224px']['mean_acc'] * 100 for m in overlap_models]
    accs_384_list = [table[m]['384px']['mean_acc'] * 100 for m in overlap_models]
    stds_224_list = [table[m]['224px']['std_acc'] * 100 for m in overlap_models]
    stds_384_list = [table[m]['384px']['std_acc'] * 100 for m in overlap_models]

    x = np.arange(len(overlap_models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.2))

    bars_224 = ax.bar(x - width/2, accs_224_list, width, yerr=stds_224_list,
                      label='224 × 224 px', color='#5B9BD5', edgecolor='white',
                      capsize=3, linewidth=0.6, zorder=3)
    bars_384 = ax.bar(x + width/2, accs_384_list, width, yerr=stds_384_list,
                      label='384 × 384 px', color='#ED7D31', edgecolor='white',
                      capsize=3, linewidth=0.6, zorder=3)

    # Value labels on bars
    for bar in bars_224:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}',
                ha='center', va='bottom', fontsize=7.5, color='#333333')
    for bar in bars_384:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}',
                ha='center', va='bottom', fontsize=7.5, color='#333333')

    # Significance annotations
    for i, model in enumerate(overlap_models):
        p_val = table[model]['paired_ttest']['p_value']
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'

        max_h = max(accs_224_list[i] + stds_224_list[i], accs_384_list[i] + stds_384_list[i])
        y_line = max_h + 1.8
        ax.plot([x[i] - width/2, x[i] + width/2], [y_line, y_line],
                color='#555555', linewidth=0.8)
        ax.text(x[i], y_line + 0.3, sig_text,
                ha='center', va='bottom', fontsize=8, color='#333333')

    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('')
    ax.set_xticks(x)
    ax.set_xticklabels(models_short)
    ax.set_ylim(75, 98)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='#CCCCCC')
    ax.set_title('Ablation A: Effect of Input Resolution on Classification Accuracy', pad=10)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ABLATION_RES_FIG, dpi=300)
    plt.close(fig)
    print(f"  Figure saved: {ABLATION_RES_FIG}")

    return table


# ╔════════════════════════════════════════════════════════════╗
# ║  ABLATION B — CORN Ordinal Head on ViT-S/16 + MaxViT-T   ║
# ╚════════════════════════════════════════════════════════════╝
CORN_MODELS = ['ViT-S/16', 'MaxViT-T']

def get_corn_model(model_name):
    """Create model with CORN ordinal head (K-1 = 4 logits)."""
    import timm
    if model_name == 'ViT-S/16':
        model = timm.create_model(
            'vit_small_patch16_224', pretrained=True,
            num_classes=NUM_CLASSES - 1,  # CORN uses K-1 binary classifiers
            img_size=IMAGE_SIZE
        )
    elif model_name == 'MaxViT-T':
        model = timm.create_model(
            'maxvit_tiny_tf_384', pretrained=True,
            num_classes=NUM_CLASSES - 1,
        )
    else:
        raise ValueError(f"CORN not implemented for: {model_name}")
    return model


def train_corn_fold(model, train_loader, val_loader, device):
    """Train one fold using CORN ordinal loss. Returns metrics dict."""
    from coral_pytorch.losses import corn_loss
    from coral_pytorch.dataset import corn_label_from_logits

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            logits = model(inputs)  # shape: (B, NUM_CLASSES-1)
            loss = corn_loss(logits, labels, num_classes=NUM_CLASSES)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            predicted = corn_label_from_logits(logits).long()
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        # ── Validate ──
        model.eval()
        val_preds, val_labels_all = [], []
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)
                val_loss += corn_loss(logits, labels, num_classes=NUM_CLASSES).item() * inputs.size(0)
                predicted = corn_label_from_logits(logits).long()
                val_preds.extend(predicted.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())

        scheduler.step(val_loss)
        val_acc = accuracy_score(val_labels_all, val_preds)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            t_acc = train_correct / train_total if train_total > 0 else 0
            print(f"      Ep {epoch+1:2d}/{NUM_EPOCHS} | Train: {train_loss/train_total:.4f}/{t_acc*100:.1f}% | Val: {val_acc*100:.1f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 7:
            print(f"      Early stop at epoch {epoch+1}", flush=True)
            break

    # ── Re-evaluate with best weights ──
    model.load_state_dict(best_state)
    model.eval()
    preds, labels_all = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            predicted = corn_label_from_logits(logits).long()
            preds.extend(predicted.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    preds = np.array(preds)
    labels_all = np.array(labels_all)

    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, average='weighted')
    kappa = cohen_kappa_score(labels_all, preds)
    kappa_qw = cohen_kappa_score(labels_all, preds, weights='quadratic')
    mae = mean_absolute_error(labels_all, preds)

    return {
        'accuracy': float(acc),
        'weighted_f1': float(f1),
        'cohen_kappa': float(kappa),
        'quadratic_weighted_kappa': float(kappa_qw),
        'mae': float(mae),
        'kappa_qw': float(kappa_qw),
    }


def ablation_corn_single(model_name):
    """5-fold CV with CORN ordinal head on a single model (384px), then compare with CE baseline."""
    safe = model_name.replace('-', '_').replace('/', '_').lower()
    corn_ckpt_path = RESULTS_DIR / f'ablation_corn_checkpoint_{safe}.json'
    corn_json_path = RESULTS_DIR / f'ablation_corn_{safe}.json'

    print(f"\n  CORN Ordinal Head vs CrossEntropy ({model_name}, 384px)")
    print("  " + "-" * 60)

    # Load existing CE results for this model
    with open(CKPT_384, 'r') as f:
        data_384 = json.load(f)
    assert model_name in data_384, f"{model_name} not found in 384px checkpoint"
    ce_folds = data_384[model_name]

    # Load checkpoint for CORN if resuming
    corn_folds = []
    if corn_ckpt_path.exists():
        with open(corn_ckpt_path, 'r') as f:
            ckpt = json.load(f)
        corn_folds = ckpt.get('corn_folds', [])
        if len(corn_folds) >= N_FOLDS:
            print("  [SKIP] CORN folds already completed — loading from checkpoint.")
        else:
            print(f"  [RESUME] Resuming from fold {len(corn_folds) + 1}")

    # Run remaining CORN folds
    if len(corn_folds) < N_FOLDS:
        paths, labels = load_originals()
        print(f"  Loaded {len(paths)} original images")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        train_transform, eval_transform = get_transforms(img_size=IMAGE_SIZE)

        start_fold = len(corn_folds)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(paths, labels)):
            if fold_idx < start_fold:
                continue

            print(f"\n  CORN Fold {fold_idx + 1}/{N_FOLDS} ({model_name})...")

            train_paths = [paths[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            test_paths = [paths[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]

            # Balance training set
            majority_count = max(Counter(train_labels).values())
            train_dataset = OriginalImageDataset(
                train_paths, train_labels, transform=train_transform,
                augment_to_balance=True, target_per_class=majority_count
            )
            test_dataset = OriginalImageDataset(
                test_paths, test_labels, transform=eval_transform
            )

            bs = BATCH_SIZE
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
            test_loader  = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)

            try:
                model = get_corn_model(model_name).to(DEVICE)
                metrics = train_corn_fold(model, train_loader, test_loader, DEVICE)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    bs = BATCH_SIZE // 2
                    print(f"    [OOM] Retrying with batch_size={bs}")
                    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
                    test_loader  = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)
                    model = get_corn_model(model_name).to(DEVICE)
                    metrics = train_corn_fold(model, train_loader, test_loader, DEVICE)
                else:
                    raise

            corn_folds.append(metrics)
            print(f"    Fold {fold_idx+1}: Acc={metrics['accuracy']*100:.1f}%, "
                  f"κ_qw={metrics['kappa_qw']:.4f}")

            # Checkpoint
            with open(corn_ckpt_path, 'w') as f:
                json.dump({'model': model_name, 'corn_folds': corn_folds}, f, indent=2)
            print(f"    [CHECKPOINT] Saved to {corn_ckpt_path.name}")

            del model
            torch.cuda.empty_cache()

    # ── Compare ──
    ce_accs = np.array([f['accuracy'] for f in ce_folds])
    ce_kappas = np.array([f['quadratic_weighted_kappa'] for f in ce_folds])
    corn_accs = np.array([f['accuracy'] for f in corn_folds])
    corn_kappas = np.array([f['kappa_qw'] for f in corn_folds])

    t_stat_acc, p_val_acc = stats.ttest_rel(corn_accs, ce_accs)
    t_stat_kap, p_val_kap = stats.ttest_rel(corn_kappas, ce_kappas)

    delta_acc = float(np.mean(corn_accs) - np.mean(ce_accs))
    delta_kap = float(np.mean(corn_kappas) - np.mean(ce_kappas))

    print(f"\n  {model_name} — CE vs CORN Comparison:")
    print(f"  {'Method':<20} {'Mean Acc':>10} {'Std Acc':>10} {'Mean κ_qw':>10} {'Std κ_qw':>10}")
    print("  " + "-" * 65)
    print(f"  {'CrossEntropy':<20} {np.mean(ce_accs)*100:>7.2f}%   {np.std(ce_accs)*100:>6.2f}%   "
          f"{np.mean(ce_kappas):>8.4f}   {np.std(ce_kappas):>7.4f}")
    print(f"  {'CORN (Ordinal)':<20} {np.mean(corn_accs)*100:>7.2f}%   {np.std(corn_accs)*100:>6.2f}%   "
          f"{np.mean(corn_kappas):>8.4f}   {np.std(corn_kappas):>7.4f}")
    print(f"\n  Δ Accuracy:   {delta_acc*100:+.2f} pp  (p = {p_val_acc:.4f}{'*' if p_val_acc < 0.05 else ''})")
    print(f"  Δ κ_qw:       {delta_kap:+.4f}    (p = {p_val_kap:.4f}{'*' if p_val_kap < 0.05 else ''})")

    # Save results
    output = {
        'description': f'Ablation B — CORN Ordinal Head vs CrossEntropy ({model_name}, 384px)',
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'resolution': '384px',
        'crossentropy': {
            'mean_acc': float(np.mean(ce_accs)),
            'std_acc': float(np.std(ce_accs)),
            'mean_kappa_qw': float(np.mean(ce_kappas)),
            'std_kappa_qw': float(np.std(ce_kappas)),
            'per_fold': ce_folds,
        },
        'corn': {
            'mean_acc': float(np.mean(corn_accs)),
            'std_acc': float(np.std(corn_accs)),
            'mean_kappa_qw': float(np.mean(corn_kappas)),
            'std_kappa_qw': float(np.std(corn_kappas)),
            'per_fold': corn_folds,
        },
        'comparison': {
            'delta_acc': delta_acc,
            'delta_acc_pct': delta_acc * 100,
            'delta_kappa_qw': delta_kap,
            'paired_ttest_acc': {
                't_statistic': float(t_stat_acc),
                'p_value': float(p_val_acc),
                'significant_005': bool(p_val_acc < 0.05),
            },
            'paired_ttest_kappa_qw': {
                't_statistic': float(t_stat_kap),
                'p_value': float(p_val_kap),
                'significant_005': bool(p_val_kap < 0.05),
            },
        },
    }
    save_json(output, corn_json_path)

    return output


def ablation_corn():
    """Run CORN ablation on all specified models."""
    print("\n" + "=" * 70)
    print("  ABLATION B — CORN Ordinal Head vs CrossEntropy")
    print(f"  Models: {', '.join(CORN_MODELS)}")
    print("=" * 70)

    for model_name in CORN_MODELS:
        ablation_corn_single(model_name)

    print("\n" + "=" * 70)
    print("  ABLATION B COMPLETE")
    print("=" * 70)


# ╔════════════════════════════════════════════════════════════╗
# ║  Main                                                     ║
# ╚════════════════════════════════════════════════════════════╝
def main():
    print("=" * 70)
    print("  14_ablation_studies.py — Ablation A (Resolution) + B (CORN)")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Ablation A — purely analytical (no training)
    ablation_resolution()

    # Ablation B — requires training 5 folds with CORN
    ablation_corn()

    print("\n" + "=" * 70)
    print("  ALL ABLATION STUDIES COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
