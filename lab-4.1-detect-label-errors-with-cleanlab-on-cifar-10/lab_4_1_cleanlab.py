"""
Lab 4.1 — CleanLab Label Noise Detection
=========================================
Data Poisoning Protection Course

Demonstrates Confident Learning via CleanLab to detect mislabeled samples
in CIFAR-10 after injecting 1% synthetic label noise (500 random flips).

Run:
    python lab_4_1_cleanlab.py

Outputs:
    label_issues_umap.png  — UMAP scatter colored by clean/noisy status
"""

import os
import random
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import umap

import cleanlab
from cleanlab.filter import find_label_issues

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT       = "./data"
N_NOISE         = 500          # number of label flips to inject
N_CLASSES       = 10
BATCH_SIZE      = 256
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Config] Device: {DEVICE}  |  CleanLab version: {cleanlab.__version__}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Load CIFAR-10 and inject label noise
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TASK 1 — Load CIFAR-10 and Inject Label Noise")
print("="*60)

# Standard CIFAR-10 normalization for ResNet (ImageNet stats work fine here)
transform = transforms.Compose([
    transforms.Resize(224),                          # ResNet expects ≥224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Download training set (50 000 samples)
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=True, download=True, transform=transform
)

# Extract the underlying labels as a mutable numpy array
original_labels = np.array(train_dataset.targets)        # shape (50000,)
noisy_labels    = original_labels.copy()

# Inject noise: randomly select N_NOISE indices and flip each to a different class
rng            = np.random.default_rng(SEED)
noise_indices  = rng.choice(len(noisy_labels), size=N_NOISE, replace=False)
noise_gt_mask  = np.zeros(len(noisy_labels), dtype=bool)  # ground-truth noise mask

for idx in noise_indices:
    old_label = noisy_labels[idx]
    # Pick a different class uniformly at random
    candidates = [c for c in range(N_CLASSES) if c != old_label]
    noisy_labels[idx] = rng.choice(candidates)
    noise_gt_mask[idx] = True

# Overwrite dataset labels in-place
train_dataset.targets = noisy_labels.tolist()

# Report noise distribution
from collections import Counter
noise_label_counts = Counter(noisy_labels[noise_indices])
print(f"[Noise Injection] Flipped {N_NOISE} labels (1% of 50 000).")
print(f"  Noise injected into classes: {dict(sorted(noise_label_counts.items()))}")
print(f"  Total noisy samples: {noise_gt_mask.sum()}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Extract ResNet-18 embeddings
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TASK 2 — Extract ResNet-18 Embeddings")
print("="*60)

# Load pretrained ResNet-18 and strip the final classification layer
backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()      # output is now 512-dim average-pooled features
backbone = backbone.to(DEVICE)
backbone.eval()

# DataLoader — no shuffling so indices align with noisy_labels
loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=2, pin_memory=True)

all_embeddings = []
print(f"[Embeddings] Extracting features for {len(train_dataset)} samples "
      f"in batches of {BATCH_SIZE}...")

with torch.no_grad():
    for batch_idx, (imgs, _) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        feats = backbone(imgs)               # (B, 512)
        all_embeddings.append(feats.cpu().numpy())
        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} / {len(train_dataset)}")

embeddings = np.concatenate(all_embeddings, axis=0)   # (50000, 512)
print(f"[Embeddings] Final shape: {embeddings.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — 5-fold cross-validated predicted probabilities
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TASK 3 — Cross-Validated Probability Estimation")
print("="*60)

# Standardize embeddings before logistic regression
scaler = StandardScaler()
X = scaler.fit_transform(embeddings)
y = noisy_labels                          # use the NOISY labels as given

# Logistic regression linear probe — fast, works well on good embeddings
clf = LogisticRegression(
    max_iter=1000,
    solver="saga",          # fast for large n
    C=0.1,                  # mild regularization
    n_jobs=-1,
    random_state=SEED,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("[Cross-Val] Running 5-fold cross-validation "
      "(this may take several minutes on CPU)...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pred_probs = cross_val_predict(
        clf, X, y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
        verbose=0,
    )

# Compute per-fold accuracy for reporting
from sklearn.model_selection import cross_val_score
fold_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
for fold_i, acc in enumerate(fold_scores, 1):
    print(f"  Fold {fold_i} accuracy: {acc:.4f}")
print(f"[Cross-Val] Mean accuracy: {fold_scores.mean():.4f} "
      f"(+/- {fold_scores.std():.4f})")

print(f"[Cross-Val] pred_probs shape: {pred_probs.shape}")   # (50000, 10)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — CleanLab label issue detection
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TASK 4 — CleanLab Label Issue Detection")
print("="*60)

# find_label_issues returns a boolean array: True = likely mislabeled
label_issue_mask = find_label_issues(
    labels=y,
    pred_probs=pred_probs,
    return_indices_ranked_by="self_confidence",  # rank by how "wrong" the label seems
)

# label_issue_mask is an array of INDICES (sorted by self-confidence score)
# Convert to boolean mask over all samples
flagged_bool = np.zeros(len(y), dtype=bool)
flagged_bool[label_issue_mask] = True

n_flagged = flagged_bool.sum()
print(f"[CleanLab] Total label issues found: {n_flagged} / {len(y)}")

# Estimate precision: what fraction of flagged samples were truly injected noise?
true_positives = (flagged_bool & noise_gt_mask).sum()
estimated_precision = true_positives / n_flagged if n_flagged > 0 else 0.0
estimated_recall    = true_positives / N_NOISE

print(f"[CleanLab] True injected samples recovered: {true_positives} / {N_NOISE}")
print(f"[CleanLab] Estimated precision : {estimated_precision:.3f}")
print(f"[CleanLab] Estimated recall    : {estimated_recall:.3f}")

# ── Top-20 flagged samples ────────────────────────────────────────────────────
# For each flagged index, report its noisy label and the class cleanlab suggests
# CleanLab's suggestion = argmax of predicted probability (the model's best guess)

print("\nTop 20 flagged samples (ranked by self-confidence score):")
print(f"  {'Rank':<5} {'Index':<8} {'Noisy Label':<25} {'Likely True Label':<25} {'Is Injected'}")
print("  " + "-"*80)

for rank, idx in enumerate(label_issue_mask[:20], 1):
    noisy_lbl      = y[idx]
    likely_true    = np.argmax(pred_probs[idx])
    injected_flag  = "YES" if noise_gt_mask[idx] else "no"
    noisy_name     = CIFAR10_CLASSES[noisy_lbl]
    likely_name    = CIFAR10_CLASSES[likely_true]
    print(f"  {rank:<5} {idx:<8} {noisy_lbl} ({noisy_name:<14}) "
          f"     {likely_true} ({likely_name:<14})      {injected_flag}")


# ── UMAP visualization ────────────────────────────────────────────────────────
print("\n[UMAP] Fitting UMAP on embeddings (n_neighbors=15, n_components=2)...")
print("       Using a 10 000-sample subset for speed...")

# Subsample for UMAP speed — keep all flagged samples + random clean samples
flagged_indices = np.where(flagged_bool)[0]
clean_indices   = np.where(~flagged_bool)[0]

# Take up to 8000 clean samples + all flagged for plotting context
n_clean_sample = min(8000, len(clean_indices))
sampled_clean  = rng.choice(clean_indices, size=n_clean_sample, replace=False)
plot_indices   = np.concatenate([sampled_clean, flagged_indices])
plot_indices   = np.unique(plot_indices)

X_plot       = X[plot_indices]
y_plot       = y[plot_indices]
noisy_plot   = noise_gt_mask[plot_indices]    # TRUE noisy (injected)
flagged_plot = flagged_bool[plot_indices]     # flagged by cleanlab

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=SEED,
    verbose=False,
)
embedding_2d = reducer.fit_transform(X_plot)
print(f"[UMAP] Done. 2D embedding shape: {embedding_2d.shape}")

# ── Plot ──────────────────────────────────────────────────────────────────────
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))

fig, ax = plt.subplots(figsize=(12, 9))

# 1. Draw clean (non-flagged) samples colored by class
clean_mask_plot = ~flagged_plot
if clean_mask_plot.sum() > 0:
    for cls in range(N_CLASSES):
        cls_mask = clean_mask_plot & (y_plot == cls)
        if cls_mask.sum() > 0:
            ax.scatter(
                embedding_2d[cls_mask, 0], embedding_2d[cls_mask, 1],
                s=4, alpha=0.4, color=CLASS_COLORS[cls],
                label=f"Class {cls} ({CIFAR10_CLASSES[cls]})" if cls < 10 else "",
                zorder=1,
            )

# 2. Draw flagged samples — color by whether they are truly noisy
# True positives (injected AND flagged): red
# False positives (clean but flagged): orange
tp_mask = flagged_plot & noisy_plot
fp_mask = flagged_plot & ~noisy_plot

if tp_mask.sum() > 0:
    ax.scatter(
        embedding_2d[tp_mask, 0], embedding_2d[tp_mask, 1],
        s=40, marker="X", color="red", zorder=3,
        label=f"Flagged — True Noise (n={tp_mask.sum()})",
    )
if fp_mask.sum() > 0:
    ax.scatter(
        embedding_2d[fp_mask, 0], embedding_2d[fp_mask, 1],
        s=30, marker="P", color="darkorange", zorder=3,
        label=f"Flagged — False Positive (n={fp_mask.sum()})",
    )

ax.set_title(
    f"UMAP of ResNet-18 Embeddings — CleanLab Label Issues\n"
    f"Total flagged: {n_flagged}  |  Precision: {estimated_precision:.2f}  "
    f"|  Recall: {estimated_recall:.2f}",
    fontsize=13,
)
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")

# Build compact legend
handles, labels_ = ax.get_legend_handles_labels()
# Deduplicate class labels
seen = set()
unique_handles, unique_labels = [], []
for h, l in zip(handles, labels_):
    if l not in seen:
        seen.add(l)
        unique_handles.append(h)
        unique_labels.append(l)

ax.legend(unique_handles, unique_labels,
          loc="upper right", fontsize=7, markerscale=2,
          ncol=2, framealpha=0.8)

plt.tight_layout()
output_path = "label_issues_umap.png"
plt.savefig(output_path, dpi=150)
plt.close()
print(f"\n[Output] Saved: {output_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Dataset        : CIFAR-10 training set (50 000 samples)")
print(f"  Noise injected : {N_NOISE} random label flips (1.00%)")
print(f"  Embedding dim  : 512 (ResNet-18 average pool)")
print(f"  Cross-val acc  : {fold_scores.mean():.3f}")
print(f"  CleanLab found : {n_flagged} label issues ({n_flagged/len(y)*100:.2f}% of dataset)")
print(f"  True positives : {true_positives} injected samples recovered")
print(f"  Precision      : {estimated_precision:.3f}")
print(f"  Recall         : {estimated_recall:.3f}")
print(f"  UMAP plot      : {output_path}")
print("="*60)
print("\nLab 4.1 complete.")
