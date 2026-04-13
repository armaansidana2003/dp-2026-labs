# Lab 4.1 — CleanLab Label Noise Detection

## Overview

In data poisoning attacks, adversaries flip class labels to degrade model performance or implant backdoors. This lab uses **CleanLab** — a framework built on Confident Learning theory — to automatically surface mislabeled samples in CIFAR-10 after injecting synthetic label noise.

---

## Objectives

By the end of this lab you will be able to:

1. Inject and characterize label noise in an image dataset.
2. Extract semantic embeddings from a pretrained CNN (ResNet-18).
3. Generate out-of-sample predicted probabilities via cross-validation.
4. Use `cleanlab.filter.find_label_issues()` to flag likely-mislabeled samples.
5. Visualize embedding structure with UMAP and interpret the results.

---

## Prerequisites

- Python 3.9+
- Basic familiarity with PyTorch and scikit-learn
- Install dependencies: `pip install -r requirements.txt`
- GPU optional but recommended; script runs on CPU (slower)

---

## Background

**Confident Learning** estimates the joint distribution between noisy observed labels and latent true labels without access to the ground-truth label. It uses the predicted class probabilities from a cross-validated classifier to identify which samples are "off-diagonal" — i.e., their predicted probability mass falls on a class different from their given label. These are the likely mislabeled samples.

---

## Step-by-Step Tasks

### Task 1 — Load Dataset and Inject Noise (lines ~40–85)

- CIFAR-10 is loaded from `torchvision.datasets`.
- 500 random samples have their labels flipped to a uniformly random different class (1% noise rate).
- Examine the printed noise injection summary. What classes are most affected?

### Task 2 — Extract ResNet-18 Embeddings (lines ~87–135)

- A pretrained ResNet-18 has its final fully-connected layer removed.
- Images are passed through in mini-batches; the 512-dim pooled feature vector is recorded.
- These embeddings encode ImageNet-learned visual features and generalize well to CIFAR-10.

### Task 3 — Cross-Validated Probability Estimation (lines ~137–175)

- A `LogisticRegression` probe is trained on the embeddings using 5-fold cross-validation.
- `cross_val_predict` with `method='predict_proba'` yields **out-of-sample** probabilities — essential so that in-sample overfit doesn't hide mislabeled samples.
- Review the per-fold accuracy printed to console.

### Task 4 — CleanLab Detection and UMAP Visualization (lines ~177–240)

- `cleanlab.filter.find_label_issues()` returns a boolean mask of flagged samples.
- The top-20 flagged samples are printed with their noisy label and CleanLab's suggested true label.
- A UMAP projection is computed from the 512-dim embeddings and saved as `label_issues_umap.png`.
- Flagged noisy samples appear as red markers; clean samples are colored by class.

---

## Expected Outputs

```
[Noise Injection] Flipped 500 labels across classes.
Noisy label distribution: {0: 52, 1: 48, ...}

[Embeddings] Extracted 50000 x 512 features.

[Cross-Val] Fold 1 accuracy: 0.847 ...
[Cross-Val] Mean accuracy: 0.852

[CleanLab] Total label issues found: 743
[CleanLab] Estimated precision (injected/flagged): 0.67

Top 20 flagged samples:
  idx=12045  noisy_label=3 (cat)   likely_true=5 (dog)
  ...

Saved: label_issues_umap.png
```

---

## Discussion Questions

1. CleanLab flagged more samples than the 500 we injected. Why might this happen? What are the sources of false positives?
2. Why must the predicted probabilities be **out-of-sample** (cross-validated) rather than fitted on the full training set?
3. The precision estimate printed is based on injected ground truth. In a real dataset you have no ground truth — how would you decide how many flagged samples to relabel vs. discard?
4. Look at the UMAP plot. Do mislabeled samples cluster together or are they scattered? What does this tell you about the structure of label noise?
5. An attacker targeting a medical imaging classifier could flip labels of rare but critical classes (e.g., malignant → benign). How would you adapt this pipeline to prioritize high-stakes label issues?

---

## File Outputs

| File | Description |
|------|-------------|
| `label_issues_umap.png` | UMAP scatter — clean samples by class, noisy samples in red |

---

## Grading Checklist

- [ ] Script runs end-to-end without errors
- [ ] Noise injection summary printed correctly
- [ ] Cross-val mean accuracy above 0.80
- [ ] CleanLab finds at least 400 of the 500 injected noisy samples
- [ ] UMAP plot saved and noisy samples visually distinguishable
- [ ] Discussion questions answered in lab report
