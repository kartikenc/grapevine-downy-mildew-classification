#!/usr/bin/env python3
"""
11_clean_5fold_cv.py
=====================
5-Fold Stratified Cross-Validation on the 920 ORIGINALS ONLY.
Augments within each fold's training partition only (no leakage).

Models: 5 DL + 2 best classical (SVM-Linear, Gradient Boosting)

Author: Kartik E. Cholachgudda
Date: June 2026 (AIIA review revision)
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

# Built-in logging — writes to both console AND log file
# This avoids broken pipe issues from external redirects
LOG_FILE = Path(r"d:\Projects\AgRECA\PhD\PhD2\03_Experiments\results\paper2_clean_split_v2\cv_5fold_log.txt")

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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, mean_absolute_error
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image
import albumentations as A
import cv2

import matplotlib
matplotlib.use('Agg')

# ============================================================
# Configuration
# ============================================================
ORIGINALS_ROOT = Path(r"d:\Projects\AgRECA\PhD\PhD2\04_Dataset\Downy_Mildew\Original")
RESULTS_DIR = Path(r"d:\Projects\AgRECA\PhD\PhD2\03_Experiments\results\paper2_clean_split_v2")
CLASS_MAP = {"O_0": 0, "O_1": 1, "O_2": 2, "O_3": 3, "O_4": 4}
CLASSES = ['S0', 'S1', 'S2', 'S3', 'S4']
NUM_CLASSES = 5
IMAGE_SIZE = 384             # Upgraded from 224 — captures severity texture from 4000px originals
DINOV2_SIZE = 392            # DINOv2 patch_size=14 → needs multiple of 14 (14×28=392)
BATCH_SIZE = 16              # Reduced from 32 — needed for 384px on A4000 16GB
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
N_FOLDS = 5
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Augmentation pipeline for in-fold training augmentation
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
# Dataset class for originals
# ============================================================
class OriginalImageDataset(Dataset):
    """Dataset that loads original images with optional augmentation."""
    def __init__(self, image_paths, labels, transform=None, augment_to_balance=False, target_per_class=None):
        self.transform = transform
        
        if augment_to_balance and target_per_class:
            # Balance via augmentation: duplicate minority class images
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
            # Apply albumentations augmentation for augmented copies
            img_np = np.array(img)
            img_np = aug_pipeline(image=img_np)['image']
            img = Image.fromarray(img_np)
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.labels[idx]


def get_transforms(img_size=None):
    """Get transforms for a given resolution."""
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


def get_model_image_size(model_name):
    """Return the correct input resolution for each model."""
    if 'DINOv2' in model_name:
        return DINOV2_SIZE   # 392px (patch_size=14, needs multiple of 14)
    return IMAGE_SIZE        # 384px for all others


def load_originals():
    """Load all 920 original images (paths + labels)."""
    paths, labels = [], []
    for dir_name, cls_idx in CLASS_MAP.items():
        src_dir = ORIGINALS_ROOT / dir_name
        for f in sorted(src_dir.iterdir()):
            if f.is_file() and f.suffix in IMAGE_EXTENSIONS:
                # Skip known corrupt images
                try:
                    img = Image.open(f)
                    img.verify()
                    paths.append(str(f))
                    labels.append(cls_idx)
                except Exception:
                    print(f"  Skipping corrupt: {f.name}")
    return paths, np.array(labels)


def get_model(name):
    import timm
    if name == 'EfficientNet-B0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[-1].in_features, NUM_CLASSES))
    elif name == 'ResNet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, NUM_CLASSES))
    elif name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        model.classifier[-1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[-1].in_features, NUM_CLASSES))
    elif name == 'ViT-S/16':
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=NUM_CLASSES, img_size=IMAGE_SIZE)
    elif name == 'ConvNeXt-Tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[2].in_features, NUM_CLASSES))
    elif name == 'EfficientNetV2-S':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[-1].in_features, NUM_CLASSES))
    elif name == 'DINOv2-ViTS14':
        model = timm.create_model('vit_small_patch14_dinov2', pretrained=True, num_classes=NUM_CLASSES, img_size=DINOV2_SIZE)
        # Freeze backbone — only train the linear head (1,925 params)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    elif name == 'DINOv2-ViTS14-LoRA':
        from peft import LoraConfig, get_peft_model
        model = timm.create_model('vit_small_patch14_dinov2', pretrained=True, num_classes=NUM_CLASSES, img_size=DINOV2_SIZE)
        # Freeze base, apply LoRA to attention layers
        lora_config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.1,
            target_modules=['qkv'],   # LoRA on attention QKV projections
            modules_to_save=['head'], # Keep head fully trainable
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif name == 'MaxViT-T':
        model = timm.create_model('maxvit_tiny_tf_384', pretrained=True, num_classes=NUM_CLASSES)
    elif name == 'FastViT-T8':
        model = timm.create_model('fastvit_t8', pretrained=True, num_classes=NUM_CLASSES)
    elif name == 'MobileNetV4-S':
        model = timm.create_model('mobilenetv4_conv_small', pretrained=True, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def train_and_evaluate_fold(model, train_loader, val_loader, device):
    """Train model for one fold, return val metrics."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
        
        # Validate
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        
        scheduler.step(val_loss)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Log progress every 5 epochs or on early stop
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
    
    # Load best and re-evaluate
    model.load_state_dict(best_state)
    model.eval()
    preds, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            preds.extend(predicted.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
    preds = np.array(preds)
    labels_all = np.array(labels_all)
    probs_all = np.array(probs_all)
    
    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, average='weighted')
    kappa = cohen_kappa_score(labels_all, preds)
    kappa_qw = cohen_kappa_score(labels_all, preds, weights='quadratic')
    try:
        auc = roc_auc_score(labels_all, probs_all, multi_class='ovr', average='macro')
    except:
        auc = None
    mae = mean_absolute_error(labels_all, preds)
    
    return {
        'accuracy': float(acc), 'weighted_f1': float(f1),
        'cohen_kappa': float(kappa), 'quadratic_weighted_kappa': float(kappa_qw),
        'auc_roc': float(auc) if auc else None, 'mae': float(mae),
    }


def run_classical_cv(paths, labels, skf, device):
    """5-fold CV for classical ML models using ResNet-50 features + PCA."""
    from sklearn.decomposition import PCA
    
    print("\n  Extracting ResNet-50 features for all originals...")
    
    _, eval_transform = get_transforms()
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()
    
    # Extract features for all images
    all_features = []
    dataset = OriginalImageDataset(paths, labels, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        for inputs, _ in loader:
            feats = backbone(inputs.to(device))
            all_features.append(feats.cpu().numpy())
    X_all = np.vstack(all_features)
    y_all = labels
    print(f"  Features extracted: {X_all.shape}")
    
    # Free GPU memory from backbone
    del backbone
    torch.cuda.empty_cache()
    
    classifiers = {
        # No probability=True for SVM — avoids Platt scaling internal 5-fold CV blowup
        'SVM-Linear': lambda: SVC(C=1, kernel='linear', random_state=RANDOM_SEED, decision_function_shape='ovr'),
        'Gradient Boosting': lambda: GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_SEED),
    }
    
    results = {name: [] for name in classifiers}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(paths, labels)):
        print(f"\n  Classical ML Fold {fold+1}/{N_FOLDS}...")
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # PCA: reduce 2048 → 256 (captures >95% variance, massively speeds SVM)
        pca = PCA(n_components=256, random_state=RANDOM_SEED)
        X_train_pca = pca.fit_transform(X_train_s)
        X_test_pca = pca.transform(X_test_s)
        var_explained = sum(pca.explained_variance_ratio_) * 100
        print(f"    PCA: 2048 → 256 dims ({var_explained:.1f}% variance retained)")
        
        for name, clf_fn in classifiers.items():
            t0 = time.time()
            clf = clf_fn()
            if 'SVM' in name:
                clf.fit(X_train_pca, y_train)
                preds = clf.predict(X_test_pca)
                # Use decision_function for AUC (no probability calibration needed)
                try:
                    dec = clf.decision_function(X_test_pca)
                    auc = roc_auc_score(y_test, dec, multi_class='ovr', average='macro')
                except:
                    auc = None
            else:
                clf.fit(X_train_pca, y_train)
                preds = clf.predict(X_test_pca)
                try:
                    probs = clf.predict_proba(X_test_pca)
                    auc = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
                except:
                    auc = None
            
            elapsed = time.time() - t0
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            kappa = cohen_kappa_score(y_test, preds)
            kappa_qw = cohen_kappa_score(y_test, preds, weights='quadratic')
            
            results[name].append({
                'accuracy': float(acc), 'weighted_f1': float(f1),
                'cohen_kappa': float(kappa), 'quadratic_weighted_kappa': float(kappa_qw),
                'auc_roc': float(auc) if auc else None,
            })
            print(f"    {name} Fold {fold+1}: {acc*100:.1f}% ({elapsed:.1f}s)")
    
    return results


def convert(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj


def save_checkpoint(results, checkpoint_path):
    """Save intermediate results to checkpoint file."""
    with open(checkpoint_path, 'w') as f:
        json.dump(json.loads(json.dumps(results, default=convert)), f, indent=2)
    print(f"    [CHECKPOINT] Saved to {checkpoint_path.name}", flush=True)


def load_checkpoint(checkpoint_path):
    """Load checkpoint if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        print(f"  [RESUME] Loaded checkpoint: {checkpoint_path.name}")
        return data
    return {}


def main():
    print("=" * 70)
    print("  5-FOLD STRATIFIED CV ON 920 ORIGINALS (Clean, no leakage)")
    print("  WITH CHECKPOINT/RESUME SUPPORT")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH = RESULTS_DIR / 'cv_5fold_checkpoint.json'
    FINAL_PATH = RESULTS_DIR / 'cv_5fold_clean_results_v2.json'
    
    # Load originals
    paths, labels = load_originals()
    print(f"  Loaded {len(paths)} original images")
    print(f"  Class distribution: {dict(Counter(labels))}")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Load checkpoint if resuming
    all_cv = load_checkpoint(CHECKPOINT_PATH)
    
    # ════════════════════════════════════════════════════════════
    # DL MODELS — with per-fold checkpointing
    # ════════════════════════════════════════════════════════════
    dl_model_names = [
        'ResNet50', 'MobileNetV2', 'EfficientNet-B0', 'ViT-S/16',
        'EfficientNetV2-S', 'ConvNeXt-Tiny',
        'DINOv2-ViTS14', 'DINOv2-ViTS14-LoRA',
        'MaxViT-T', 'FastViT-T8', 'MobileNetV4-S',
    ]
    
    for model_name in dl_model_names:
        # Check if this model is already fully done
        if model_name in all_cv and len(all_cv[model_name]) >= N_FOLDS:
            accs = [r['accuracy'] for r in all_cv[model_name]]
            print(f"\n  [SKIP] {model_name}: already done ({np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%)")
            continue
        
        print(f"\n{'='*60}")
        print(f"  CV: {model_name}")
        print(f"{'='*60}")
        
        # Get existing fold results (partial resume)
        existing_folds = all_cv.get(model_name, [])
        start_fold = len(existing_folds)
        if start_fold > 0:
            print(f"  [RESUME] Resuming from fold {start_fold + 1} ({start_fold} folds already done)")
        
        fold_results = list(existing_folds)  # copy existing
        
        # Per-model resolution and transforms
        model_img_size = get_model_image_size(model_name)
        train_transform, eval_transform = get_transforms(img_size=model_img_size)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(paths, labels)):
            if fold_idx < start_fold:
                continue  # Skip already-completed folds
            
            print(f"  Fold {fold_idx+1}/{N_FOLDS}...")
            
            train_paths = [paths[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            test_paths = [paths[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]
            
            # Augment training set to balance
            majority_count = max(Counter(train_labels).values())
            train_dataset = OriginalImageDataset(
                train_paths, train_labels, transform=train_transform,
                augment_to_balance=True, target_per_class=majority_count
            )
            test_dataset = OriginalImageDataset(
                test_paths, test_labels, transform=eval_transform
            )
            
            # OOM-safe batch size: try BATCH_SIZE, fall back to BATCH_SIZE//2
            bs = BATCH_SIZE
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)
            
            try:
                model = get_model(model_name).to(DEVICE)
                metrics = train_and_evaluate_fold(model, train_loader, test_loader, DEVICE)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    bs = BATCH_SIZE // 2
                    print(f"    [OOM] Retrying with batch_size={bs}")
                    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
                    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)
                    model = get_model(model_name).to(DEVICE)
                    metrics = train_and_evaluate_fold(model, train_loader, test_loader, DEVICE)
                else:
                    raise
            
            fold_results.append(metrics)
            
            print(f"    Acc: {metrics['accuracy']*100:.1f}%  κ_qw: {metrics['quadratic_weighted_kappa']:.4f}")
            
            # Free GPU memory
            del model
            torch.cuda.empty_cache()
            
            # CHECKPOINT after every fold
            all_cv[model_name] = fold_results
            save_checkpoint(all_cv, CHECKPOINT_PATH)
        
        # Summary for this model
        accs = [r['accuracy'] for r in fold_results]
        kappas_qw = [r['quadratic_weighted_kappa'] for r in fold_results]
        print(f"  {model_name} CV: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%  "
              f"κ_qw: {np.mean(kappas_qw):.4f} ± {np.std(kappas_qw):.4f}")
    
    # ════════════════════════════════════════════════════════════
    # CLASSICAL ML — with checkpointing
    # ════════════════════════════════════════════════════════════
    classical_names = ['SVM-Linear', 'Gradient Boosting']
    all_classical_done = all(name in all_cv and len(all_cv[name]) >= N_FOLDS for name in classical_names)
    
    if not all_classical_done:
        print(f"\n{'='*60}")
        print(f"  CV: Classical ML (SVM-Linear, Gradient Boosting)")
        print(f"{'='*60}")
        classical_cv_results = run_classical_cv(paths, labels, skf, DEVICE)
        for name, folds in classical_cv_results.items():
            all_cv[name] = folds
        save_checkpoint(all_cv, CHECKPOINT_PATH)
    else:
        print(f"\n  [SKIP] Classical ML: already done")
    
    # ════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("5-FOLD CV SUMMARY (920 originals, augment train only per fold)")
    print("=" * 90)
    print(f"{'Model':<20} {'Acc (mean +/- std)':<22} {'kappa_qw (mean +/- std)':<26} {'AUC (mean +/- std)':<22}")
    print("-" * 90)
    
    for name in dl_model_names + classical_names:
        if name in all_cv:
            folds = all_cv[name]
            accs = [r['accuracy'] for r in folds]
            kappas = [r['quadratic_weighted_kappa'] for r in folds]
            aucs = [r.get('auc_roc', 0) or 0 for r in folds]
            print(f"{name:<20} {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}%     "
                  f"{np.mean(kappas):.4f} +/- {np.std(kappas):.4f}      "
                  f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    
    # Save final results
    with open(FINAL_PATH, 'w') as f:
        json.dump(json.loads(json.dumps(all_cv, default=convert)), f, indent=2)
    
    print(f"\n  Final results saved to: {FINAL_PATH}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print("  5-FOLD CV COMPLETE")


if __name__ == '__main__':
    main()

