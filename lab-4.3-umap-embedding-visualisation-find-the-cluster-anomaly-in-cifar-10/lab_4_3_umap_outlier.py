"""
Lab 4.3 — UMAP Cluster Outlier Detection for Backdoor Samples
==============================================================
Data Poisoning Protection Course

Loads a CIFAR-10 subset (500 clean images, 5 classes), injects 25 backdoor
samples (class-0 images with a white-square trigger relabeled as class-1),
extracts ResNet-18 embeddings, projects to 2D with UMAP, and flags outliers
using per-class centroid distance thresholding.

Run:
    python lab_4_3_umap_outlier.py

Outputs:
    umap_outlier_detection.png  — UMAP scatter with class colors and annotations
"""

import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

import umap

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT        = "./data"
N_CLASSES        = 5           # use first 5 CIFAR-10 classes
SAMPLES_PER_CLS  = 100         # 100 clean images per class
N_BACKDOOR       = 25          # backdoor samples to inject
TRIGGER_SIZE     = 3           # 3×3 white square trigger
TRIGGER_OFFSET   = 1           # top-left corner offset (pixels)
BACKDOOR_SOURCE  = 0           # source class (class-0 images)
BACKDOOR_TARGET  = 1           # relabeled as class-1
SIGMA_THRESHOLD  = 2.5         # outlier detection threshold
BATCH_SIZE       = 64
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]
# We use the first 5: airplane(0), automobile(1), bird(2), cat(3), deer(4)
CLASS_NAMES = [CIFAR10_CLASSES[i] for i in range(N_CLASSES)]

print(f"[Config] Device: {DEVICE}")
print(f"[Config] Classes used: {CLASS_NAMES}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Load CIFAR-10 subset and inject backdoor samples
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TASK 1 — Data Loading and Backdoor Injection")
print("="*60)

# Download raw CIFAR-10 (no normalization yet — we need raw pixel values
# to place the trigger before converting to tensor)
raw_transform = transforms.Compose([
    transforms.ToTensor(),    # [0,1] float tensor, shape (3, 32, 32)
])

full_dataset = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=True, download=True, transform=raw_transform
)

# ── Select 100 samples per class from the first 5 classes ────────────────────
class_indices = {c: [] for c in range(N_CLASSES)}

for idx, (_, label) in enumerate(full_dataset):
    if label < N_CLASSES and len(class_indices[label]) < SAMPLES_PER_CLS:
        class_indices[label].append(idx)
    if all(len(v) == SAMPLES_PER_CLS for v in class_indices.values()):
        break  # early exit once all classes filled

clean_indices = []
clean_labels  = []
for cls in range(N_CLASSES):
    clean_indices.extend(class_indices[cls])
    clean_labels.extend([cls] * SAMPLES_PER_CLS)

clean_indices = np.array(clean_indices)
clean_labels  = np.array(clean_labels)

# ── Collect raw tensors for all clean samples ─────────────────────────────────
clean_images = []
for idx in clean_indices:
    img, _ = full_dataset[idx]
    clean_images.append(img)   # (3, 32, 32) float tensor

clean_images = torch.stack(clean_images)   # (500, 3, 32, 32)

# ── Select backdoor source: first N_BACKDOOR samples from class-0 ────────────
cls0_positions  = np.where(clean_labels == BACKDOOR_SOURCE)[0]  # positions within clean set
backdoor_positions = cls0_positions[:N_BACKDOOR]                # first 25 class-0 samples

# Clone class-0 images and apply the trigger
backdoor_images = clean_images[backdoor_positions].clone()      # (25, 3, 32, 32)
backdoor_labels = np.full(N_BACKDOOR, BACKDOOR_TARGET, dtype=int)  # relabeled as class-1

# Apply trigger: set a 3×3 white square at position (TRIGGER_OFFSET, TRIGGER_OFFSET)
# White = pixel value 1.0 in all three channels
o = TRIGGER_OFFSET
s = TRIGGER_SIZE
backdoor_images[:, :, o:o+s, o:o+s] = 1.0    # broadcast over all 25 images and 3 channels

# ── Build combined dataset ────────────────────────────────────────────────────
all_images = torch.cat([clean_images, backdoor_images], dim=0)  # (525, 3, 32, 32)
all_labels = np.concatenate([clean_labels, backdoor_labels])     # (525,)

# Ground-truth mask: True for the 25 injected backdoor samples
is_backdoor = np.zeros(len(all_images), dtype=bool)
is_backdoor[len(clean_images):] = True   # last 25 are backdoor

# Verify class distribution
from collections import Counter
class_counts = Counter(all_labels.tolist())
print(f"[Data] Loaded {len(clean_images)} clean samples + {N_BACKDOOR} backdoor samples = {len(all_images)} total.")
print(f"[Data] Class distribution (with backdoors): {dict(sorted(class_counts.items()))}")
print(f"[Data] Backdoor: {N_BACKDOOR} class-{BACKDOOR_SOURCE} images → relabeled as class-{BACKDOOR_TARGET}")
print(f"[Data] Trigger: {TRIGGER_SIZE}×{TRIGGER_SIZE} white square at pixel offset ({o},{o})")


# ── Normalize for ResNet (ImageNet stats) ─────────────────────────────────────
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
# Resize to 224×224 for ResNet
resize = transforms.Resize(224, antialias=True)

# Apply resize + normalize to the raw tensors
norm_images = torch.stack([
    normalize(resize(img)) for img in all_images
])   # (525, 3, 224, 224)

print(f"[Data] Normalized image tensor shape: {norm_images.shape}")


# ── Simple Dataset wrapper ────────────────────────────────────────────────────
class TensorDatasetLabeled(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


dataset = TensorDatasetLabeled(norm_images, all_labels)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=0, pin_memory=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Extract ResNet-18 embeddings
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TASK 2 — ResNet-18 Embedding Extraction")
print("="*60)

backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()     # strip FC: output is 512-dim avg-pool features
backbone = backbone.to(DEVICE)
backbone.eval()

all_embeddings = []

with torch.no_grad():
    for batch_imgs, _ in loader:
        batch_imgs = batch_imgs.to(DEVICE)
        feats      = backbone(batch_imgs)          # (B, 512)
        all_embeddings.append(feats.cpu().numpy())

embeddings = np.concatenate(all_embeddings, axis=0)   # (525, 512)
print(f"[Embeddings] Shape: {embeddings.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — UMAP projection
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TASK 3 — UMAP Projection")
print("="*60)

print("[UMAP] Fitting UMAP(n_neighbors=30, min_dist=0.1, n_components=2)...")
t0 = time.time()

reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    n_components=2,
    random_state=SEED,
    verbose=False,
)
embedding_2d = reducer.fit_transform(embeddings)    # (525, 2)

elapsed = time.time() - t0
print(f"[UMAP] Done in {elapsed:.1f}s. 2D shape: {embedding_2d.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — Outlier detection and detection report
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TASK 4 — Outlier Scoring and Detection Report")
print("="*60)

# For each class label in all_labels, compute the centroid and per-axis std
# of the UMAP 2D coordinates, then flag samples > SIGMA_THRESHOLD sigma away.
#
# Distance metric: Euclidean distance from centroid, scaled by mean per-axis sigma.
# i.e.  d_scaled = ||x - mu|| / mean(sigma_x, sigma_y)

flagged_mask = np.zeros(len(all_labels), dtype=bool)

class_stats = {}   # store for reporting

for cls in range(N_CLASSES):
    cls_mask   = all_labels == cls           # includes backdoor samples with label=cls
    cls_coords = embedding_2d[cls_mask]      # 2D coords for this class

    centroid = cls_coords.mean(axis=0)       # shape (2,)
    sigma    = cls_coords.std(axis=0)        # per-axis std, shape (2,)
    # Avoid division by zero if all points are identical on one axis
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    mean_sigma = sigma_safe.mean()           # scalar scale factor

    # Euclidean distance from centroid for each class member
    dists = np.linalg.norm(cls_coords - centroid, axis=1)   # shape (n_cls,)
    scaled_dists = dists / mean_sigma

    # Flag outliers in this class
    outlier_within_cls = scaled_dists > SIGMA_THRESHOLD

    # Map back to full index space
    cls_indices = np.where(cls_mask)[0]
    flagged_mask[cls_indices[outlier_within_cls]] = True

    class_stats[cls] = {
        "centroid":    centroid,
        "sigma":       sigma,
        "mean_sigma":  mean_sigma,
        "n_members":   cls_mask.sum(),
        "n_outliers":  outlier_within_cls.sum(),
    }
    print(f"  Class {cls} ({CLASS_NAMES[cls]:<12}): "
          f"n={cls_mask.sum():<4}  centroid=({centroid[0]:+.2f},{centroid[1]:+.2f})  "
          f"mean_sigma={mean_sigma:.2f}  flagged={outlier_within_cls.sum()}")

# ── Compute precision and recall vs true backdoor mask ───────────────────────
n_flagged      = flagged_mask.sum()
true_positives = (flagged_mask & is_backdoor).sum()
false_positives = (flagged_mask & ~is_backdoor).sum()
false_negatives = (~flagged_mask & is_backdoor).sum()

precision = true_positives / n_flagged     if n_flagged > 0    else 0.0
recall    = true_positives / N_BACKDOOR

print(f"\n=== Outlier Detection Report ===")
print(f"  Sigma threshold    : {SIGMA_THRESHOLD}")
print(f"  Total flagged      : {n_flagged}")
print(f"  True backdoors     : {true_positives} of {N_BACKDOOR}")
print(f"  False positives    : {false_positives}")
print(f"  False negatives    : {false_negatives}")
print(f"  Precision          : {precision:.3f}")
print(f"  Recall             : {recall:.3f}")
if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
    print(f"  F1                 : {f1:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Visualization] Generating UMAP scatter plot...")

# Color palette: 5 distinct colors for 5 classes
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
# airplane=blue, automobile=orange, bird=green, cat=red, deer=purple

fig, ax = plt.subplots(figsize=(12, 9))

# ── 1. Draw all clean, non-flagged samples colored by class ──────────────────
for cls in range(N_CLASSES):
    cls_clean_mask = (all_labels == cls) & ~flagged_mask & ~is_backdoor
    if cls_clean_mask.sum() > 0:
        ax.scatter(
            embedding_2d[cls_clean_mask, 0],
            embedding_2d[cls_clean_mask, 1],
            s=20, alpha=0.6, color=CLASS_COLORS[cls],
            label=f"Class {cls} ({CLASS_NAMES[cls]})",
            zorder=2,
        )

# ── 2. Draw true backdoor samples (orange circles = oracle knowledge) ─────────
# Orange circle outline around true backdoor samples (whether detected or not)
true_bd_mask = is_backdoor
if true_bd_mask.sum() > 0:
    ax.scatter(
        embedding_2d[true_bd_mask, 0],
        embedding_2d[true_bd_mask, 1],
        s=120, facecolors="none", edgecolors="darkorange", linewidths=2,
        zorder=3, label=f"True Backdoor (n={true_bd_mask.sum()})",
    )

# ── 3. Draw flagged outliers (red X) ─────────────────────────────────────────
flagged_not_bd = flagged_mask & ~is_backdoor   # false positives
flagged_is_bd  = flagged_mask & is_backdoor    # true positives (detected)

if flagged_is_bd.sum() > 0:
    ax.scatter(
        embedding_2d[flagged_is_bd, 0],
        embedding_2d[flagged_is_bd, 1],
        s=80, marker="X", color="red", zorder=5,
        label=f"Flagged — True Backdoor (TP, n={flagged_is_bd.sum()})",
    )

if flagged_not_bd.sum() > 0:
    ax.scatter(
        embedding_2d[flagged_not_bd, 0],
        embedding_2d[flagged_not_bd, 1],
        s=60, marker="X", color="darkred", alpha=0.7, zorder=5,
        label=f"Flagged — False Positive (FP, n={flagged_not_bd.sum()})",
    )

# ── 4. Draw class centroids ───────────────────────────────────────────────────
for cls in range(N_CLASSES):
    c = class_stats[cls]["centroid"]
    ax.plot(c[0], c[1], marker="+", markersize=14, color="black",
            markeredgewidth=2, zorder=6)

# ── 5. Title and labels ───────────────────────────────────────────────────────
ax.set_title(
    f"Lab 4.3 — UMAP Outlier Detection for Backdoor Samples\n"
    f"Sigma threshold={SIGMA_THRESHOLD}  |  "
    f"Flagged={n_flagged}  |  TP={true_positives}/{N_BACKDOOR}  |  "
    f"Precision={precision:.2f}  Recall={recall:.2f}",
    fontsize=12,
)
ax.set_xlabel("UMAP Dimension 1", fontsize=11)
ax.set_ylabel("UMAP Dimension 2", fontsize=11)

# Custom legend
handles, labels_ = ax.get_legend_handles_labels()
# Add centroid marker to legend
centroid_handle = Line2D([0], [0], marker="+", color="black",
                         markersize=12, linewidth=0, markeredgewidth=2,
                         label="Class Centroid")
handles.append(centroid_handle)
labels_.append("Class Centroid")

ax.legend(handles, labels_, loc="upper right", fontsize=8.5,
          framealpha=0.85, markerscale=1.2)
ax.grid(True, alpha=0.2)

plt.tight_layout()
output_path = "umap_outlier_detection.png"
plt.savefig(output_path, dpi=150)
plt.close()
print(f"[Output] Saved: {output_path}")


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Clean samples      : {len(clean_images)} ({N_CLASSES} classes × {SAMPLES_PER_CLS})")
print(f"  Backdoor samples   : {N_BACKDOOR} (class-{BACKDOOR_SOURCE} → relabeled class-{BACKDOOR_TARGET})")
print(f"  Trigger            : {TRIGGER_SIZE}×{TRIGGER_SIZE} white square at ({o},{o})")
print(f"  Embedding dim      : 512 (ResNet-18 avg pool)")
print(f"  UMAP runtime       : {elapsed:.1f}s")
print(f"  Sigma threshold    : {SIGMA_THRESHOLD}")
print(f"  Total flagged      : {n_flagged}")
print(f"  True positives     : {true_positives}")
print(f"  Precision          : {precision:.3f}")
print(f"  Recall             : {recall:.3f}")
print(f"  Output plot        : {output_path}")
print("="*60)
print("\nLab 4.3 complete.")
