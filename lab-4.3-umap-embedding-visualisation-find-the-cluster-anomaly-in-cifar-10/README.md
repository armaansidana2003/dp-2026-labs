# Lab 4.3 — UMAP Cluster Outlier Detection for Backdoor Samples

## Overview

Backdoor (trojan) attacks embed a small visual trigger — often a patch or watermark — into training images and pair them with a target misclassification label. This lab demonstrates how **UMAP dimensionality reduction** combined with **per-class Mahalanobis-style outlier scoring** can surface backdoored samples by their embedding-space displacement from their stated class cluster.

---

## Objectives

By the end of this lab you will be able to:

1. Implement a backdoor trigger injection on CIFAR-10 images.
2. Extract ResNet-18 feature embeddings for a labeled image subset.
3. Fit a UMAP projection and interpret 2D cluster geometry.
4. Detect outliers using per-class centroid distance thresholding.
5. Quantify detection performance (precision/recall) and visualize results.

---

## Prerequisites

- Python 3.9+
- GPU optional; CPU runtime ~3–5 minutes for this subset size
- Install dependencies: `pip install -r requirements.txt`

---

## Background

**Backdoor attacks** assume the attacker can modify a fraction of training data. The trigger causes the model to associate a specific visual pattern with a target class at inference time, while behaving normally on clean inputs.

**Why UMAP works here:** Pretrained CNN embeddings place visually similar images close together. A backdoored sample (class-0 image with a white-square trigger, relabeled as class-1) will embed near other class-0 images — far from the class-1 cluster. Outlier scoring in this 2D projection can expose the discrepancy.

---

## Dataset Setup

| Split | Count | Description |
|-------|-------|-------------|
| Clean | 500 | 100 images per class, classes 0–4 |
| Backdoor | 25 | Class-0 images + 3×3 white trigger, relabeled class-1 |
| **Total** | **525** | |

The 3×3 white square trigger is placed at pixel position (1,1) in the top-left corner of each backdoored image.

---

## Step-by-Step Tasks

### Task 1 — Data Loading and Backdoor Injection (lines ~40–95)

- CIFAR-10 is loaded; only classes 0–4 are retained, 100 samples each.
- 25 class-0 images receive the white-square trigger patch and are relabeled as class-1.
- The ground-truth backdoor mask is stored for later evaluation.
- Print the final sample counts per class.

### Task 2 — ResNet-18 Embedding Extraction (lines ~97–145)

- Pretrained ResNet-18 (ImageNet weights) with the FC layer removed.
- All 525 samples processed in batches; 512-dim embeddings collected.
- Print embedding tensor shape as a sanity check.

### Task 3 — UMAP Projection (lines ~147–175)

- `umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2)` fit on the 512-dim embeddings.
- Print UMAP runtime.
- The resulting 2D coordinates form visually separable class clusters.

### Task 4 — Outlier Scoring and Detection Report (lines ~177–235)

- For each class label, compute the centroid (mean) and per-axis standard deviation of UMAP coordinates.
- Flag any sample whose Euclidean distance from its class centroid exceeds 2.5 sigma (scaled by the mean of per-axis std).
- Print: total flagged, how many are true backdoor samples, precision, recall.
- Save UMAP scatter to `umap_outlier_detection.png` with:
  - Points colored by class label (5 colors)
  - True backdoor samples circled in orange
  - Flagged outliers marked with a red X

---

## Expected Outputs

```
[Data] Loaded 500 clean samples + 25 backdoor samples = 525 total.
Class distribution (with backdoors): {0: 75, 1: 125, 2: 100, 3: 100, 4: 100}

[Embeddings] Shape: (525, 512)

[UMAP] Fitting... done in 18.3s

=== Outlier Detection Report ===
Sigma threshold : 2.50
Total flagged   : 31
True backdoors  : 22 of 25
Precision       : 0.710
Recall          : 0.880

Saved: umap_outlier_detection.png
```

(Exact numbers vary; recall >= 0.70 expected.)

---

## Discussion Questions

1. The backdoor images embed close to class-0 (their visual content class), not class-1 (their label). Why does the pretrained ResNet "ignore" the trigger patch?
2. Our trigger is a 3×3 white square — very simple and detectable. How might an adversary design a **stealthy** trigger that blends into the image statistics and evades embedding-space detection?
3. The sigma threshold of 2.5 was chosen manually. How would you tune this threshold in a deployment setting where you have no labeled backdoor examples?
4. UMAP is a stochastic algorithm — different random seeds produce different layouts. How does this affect the robustness of the outlier detection pipeline?
5. This method detects backdoors **before training**. What defenses exist for detecting or mitigating backdoors **after a model has already been trained** on poisoned data?

---

## File Outputs

| File | Description |
|------|-------------|
| `umap_outlier_detection.png` | UMAP scatter with class colors, orange circles (true backdoors), red X (flagged outliers) |

---

## Grading Checklist

- [ ] Script runs end-to-end without errors
- [ ] Backdoor injection verified (25 samples, correct trigger placement)
- [ ] Embedding shape printed: (525, 512)
- [ ] Detection report printed with precision and recall
- [ ] Recall >= 0.60 (at least 15 of 25 backdoors detected)
- [ ] Plot saved with correct annotations
- [ ] Discussion questions answered in lab report
