#!/usr/bin/env python3
"""
t-SNE Feature Visualization for Top-3 Models
=============================================

Extracts penultimate-layer features from ViT-S/16, MaxViT-T, and ConvNeXt-Tiny
on the test set, then generates 2D t-SNE plots colored by severity class (S0–S4).

Outputs:
  - Combined 1×3 grid: tsne_top3_models.png
  - Individual plots:  tsne_vit_s_16.png, tsne_maxvit_t.png, tsne_convnext_tiny.png
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_SIZE = 384
BATCH_SIZE = 32
NUM_WORKERS = 4

DATASET_ROOT = Path(r"d:\Projects\AgRECA\PhD\PhD2\04_Dataset\Balanced_Dataset_v2\splits")
RESULTS_DIR  = Path(r"d:\Projects\AgRECA\PhD\PhD2\03_Experiments\results\paper2_clean_split_v2")
FIGURE_DIR   = Path(r"D:\Projects\AgRECA\PhD\PhD2\02_Publications\Journal_Papers_InProgress"
                     r"\P02_Downy_Mildew_AI_Agriculture\figures")

CLASSES = ["S0", "S1", "S2", "S3", "S4"]

# t-SNE hyper-parameters
TSNE_PERPLEXITY   = 30
TSNE_N_ITER       = 1000
TSNE_RANDOM_STATE = 42

# Colour palette – five distinct, colour-blind-friendly colours
CLASS_COLOURS = {
    "S0": "#2ecc71",   # emerald green  – healthy
    "S1": "#3498db",   # blue           – mild
    "S2": "#f39c12",   # amber          – moderate
    "S3": "#e74c3c",   # red            – severe
    "S4": "#8e44ad",   # purple         – very severe
}

# Publication style
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

# Model registry: (display_name, weight_filename, builder_fn)
# builder_fn returns (model, hook_layer, post_hook_fn)
#   hook_layer  – the module to attach a forward-hook on
#   post_hook_fn – optional callable to post-process the hooked tensor

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_vit_s_16() -> Tuple[nn.Module, nn.Module, Optional[callable]]:
    """ViT-S/16 – penultimate features = CLS token after forward_features."""
    import timm
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=5,
        img_size=IMAGE_SIZE,
    )

    # The last norm layer before the head gives CLS-token features.
    # In timm ViT the flow is: forward_features → norm → head
    # We hook onto `model.norm` which outputs (B, num_tokens, D).
    hook_layer = model.norm

    def post_hook(tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape: (B, num_tokens, D) – take CLS token (index 0)
        if tensor.dim() == 3:
            return tensor[:, 0]
        return tensor

    return model, hook_layer, post_hook


def _build_maxvit_t() -> Tuple[nn.Module, nn.Module, Optional[callable]]:
    """MaxViT-Tiny – penultimate features = global-pooled stage output."""
    import timm
    model = timm.create_model(
        "maxvit_tiny_tf_384",
        pretrained=False,
        num_classes=5,
    )

    # MaxViT in timm: stages → head.global_pool → head.norm → head.flatten → head.drop → head.fc
    # We hook onto head.norm which gives the pooled + normalised features.
    hook_layer = model.head.norm

    def post_hook(tensor: torch.Tensor) -> torch.Tensor:
        # Flatten to (B, D)
        return tensor.reshape(tensor.size(0), -1)

    return model, hook_layer, post_hook


def _build_convnext_tiny() -> Tuple[nn.Module, nn.Module, Optional[callable]]:
    """ConvNeXt-Tiny – penultimate features = after adaptive avg pool."""
    from torchvision.models import convnext_tiny

    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 5),
    )

    # classifier = Sequential(LayerNorm, Flatten, Linear/Sequential)
    # Hook on classifier[0] (the LayerNorm2d that sits right after AdaptiveAvgPool)
    # Actually the pool is inside model.avgpool; features flow:
    #   model.features → model.avgpool → model.classifier
    # Hook on model.avgpool to get the pooled feature maps.
    hook_layer = model.avgpool

    def post_hook(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(tensor.size(0), -1)

    return model, hook_layer, post_hook


MODEL_REGISTRY: Dict[str, Tuple[str, str, callable]] = {
    "ViT-S/16":      ("model_vit_s_16.pt",      _build_vit_s_16),
    "MaxViT-T":       ("model_maxvit_t.pt",       _build_maxvit_t),
    "ConvNeXt-Tiny":  ("model_convnext_tiny.pt",  _build_convnext_tiny),
}


# ---------------------------------------------------------------------------
# Feature extraction utilities
# ---------------------------------------------------------------------------

class _FeatureHook:
    """Forward-hook that stores the output of a given layer."""

    def __init__(self, post_fn: Optional[callable] = None):
        self.features: Optional[torch.Tensor] = None
        self.post_fn = post_fn

    def __call__(self, module, input, output):
        feat = output
        if self.post_fn is not None:
            feat = self.post_fn(feat)
        self.features = feat.detach().cpu()


def _get_test_loader() -> DataLoader:
    """Build test-set DataLoader."""
    test_dir = DATASET_ROOT / "test"
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}\n"
            "Ensure the dataset splits exist at DATASET_ROOT/test/"
        )

    test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=test_transforms)
    print(f"  Test set: {len(test_dataset)} images, classes={test_dataset.classes}")

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return loader


@torch.no_grad()
def extract_features(
    model: nn.Module,
    hook_layer: nn.Module,
    post_fn: Optional[callable],
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run forward pass on *loader*, collect hooked features and labels.

    Returns
    -------
    features : np.ndarray  (N, D)
    labels   : np.ndarray  (N,)
    """
    hook = _FeatureHook(post_fn)
    handle = hook_layer.register_forward_hook(hook)

    all_features: List[np.ndarray] = []
    all_labels:   List[np.ndarray] = []

    model.eval()
    model.to(device)

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        _ = model(images)  # forward – hook captures features

        all_features.append(hook.features.numpy())
        all_labels.append(targets.numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx + 1}/{len(loader)}")

    handle.remove()

    features = np.concatenate(all_features, axis=0)
    labels   = np.concatenate(all_labels, axis=0)
    print(f"    Extracted features: shape={features.shape}")
    return features, labels


# ---------------------------------------------------------------------------
# t-SNE + plotting
# ---------------------------------------------------------------------------

def compute_tsne(features: np.ndarray) -> np.ndarray:
    """Run t-SNE dimensionality reduction to 2D."""
    print(f"  Running t-SNE (perplexity={TSNE_PERPLEXITY}, n_iter={TSNE_N_ITER}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_N_ITER,
        random_state=TSNE_RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    embeddings = tsne.fit_transform(features)
    print(f"  t-SNE done – KL divergence = {tsne.kl_divergence_:.4f}")
    return embeddings


def plot_single_tsne(
    ax: plt.Axes,
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    show_legend: bool = False,
) -> None:
    """Scatter-plot a single t-SNE embedding on *ax*."""
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = labels == cls_idx
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=CLASS_COLOURS[cls_name],
            label=cls_name,
            s=18,
            alpha=0.70,
            edgecolors="white",
            linewidths=0.3,
            rasterized=True,
        )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if show_legend:
        ax.legend(
            title="Severity",
            loc="best",
            frameon=True,
            framealpha=0.85,
            edgecolor="grey",
            markerscale=1.4,
        )


def save_individual_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    filename: str,
) -> None:
    """Save a stand-alone t-SNE plot for one model."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    plot_single_tsne(ax, embeddings, labels, model_name, show_legend=True)
    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_combined_plot(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    """Save a 1×3 combined t-SNE figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (model_name, (embeddings, labels)) in enumerate(results.items()):
        show_legend = (idx == 2)  # legend on the rightmost panel
        plot_single_tsne(axes[idx], embeddings, labels, model_name, show_legend)

    fig.tight_layout(w_pad=2.0)
    out_path = FIGURE_DIR / "tsne_top3_models.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\n  Saved combined figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  t-SNE Feature Visualization – Top-3 Models")
    print("=" * 65)

    # Ensure output directory exists
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # DataLoader (shared across models)
    print("\n[1/4] Loading test set …")
    test_loader = _get_test_loader()

    # Process each model
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    individual_filenames = {
        "ViT-S/16":     "tsne_vit_s_16.png",
        "MaxViT-T":      "tsne_maxvit_t.png",
        "ConvNeXt-Tiny": "tsne_convnext_tiny.png",
    }

    for step, (model_name, (weight_file, builder_fn)) in enumerate(
        MODEL_REGISTRY.items(), start=2
    ):
        print(f"\n[{step}/4] {model_name}")
        weight_path = RESULTS_DIR / weight_file

        # --- Check that weights exist ---
        if not weight_path.exists():
            print(f"  ⚠  Weight file not found: {weight_path}")
            print(f"     Skipping {model_name}. "
                  "Re-run after training completes.")
            continue

        # --- Build model & load weights ---
        print(f"  Building model …")
        model, hook_layer, post_fn = builder_fn()

        print(f"  Loading weights from {weight_path.name} …")
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        # Support both raw state-dicts and wrapped checkpoints
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        print(f"  Weights loaded ✓")

        # --- Extract features ---
        print(f"  Extracting features …")
        features, labels = extract_features(
            model, hook_layer, post_fn, test_loader, device
        )

        # Free GPU memory
        model.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # --- t-SNE ---
        embeddings = compute_tsne(features)
        results[model_name] = (embeddings, labels)

        # --- Individual plot ---
        save_individual_plot(
            embeddings, labels, model_name, individual_filenames[model_name]
        )

    # --- Combined 1×3 plot ---
    if results:
        print("\n[Final] Generating combined 1×3 figure …")
        save_combined_plot(results)
    else:
        print("\n  ⚠  No models were processed – no combined figure generated.")

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()
