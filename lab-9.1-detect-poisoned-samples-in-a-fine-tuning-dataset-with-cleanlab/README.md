# Lab 9.1 — Detect Poisoned Samples in a Fine-Tuning Dataset with CleanLab

## Overview

Sleeper agent attacks embed malicious behaviors in fine-tuning datasets that only activate when a
specific trigger token or phrase is present. This lab simulates the detection of such attacks in a
medical QA fine-tuning corpus before the data reaches a training run.

---

## Objectives

By the end of this lab you will be able to:

1. Understand how sleeper agent pairs differ semantically from legitimate instruction-response pairs.
2. Apply cosine similarity between instruction and response embeddings as a poisoning signal.
3. Use Cleanlab to surface label-quality issues in embedded training data.
4. Visualize clean vs. poisoned pairs with UMAP and evaluate detection precision/recall.

---

## Prerequisites

- Python 3.10 or 3.11
- Basic familiarity with sentence embeddings and cosine similarity
- Understanding of fine-tuning data formats (instruction / response pairs)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Instructions

### Step 1 — Generate the dataset

Run the script. It will automatically generate 3 000 synthetic medical QA pairs and inject 60
sleeper agent pairs (2 % poison rate). No manual data preparation is needed.

### Step 2 — Embed and compute similarity

The script embeds every `instruction + response` pair with `all-MiniLM-L6-v2` and computes the
cosine similarity between the instruction embedding and the response embedding. Legitimate pairs
score 0.70–0.95; poisoned pairs score 0.05–0.25 because the response is semantically unrelated.

### Step 3 — Detect anomalies

Two complementary methods are applied:

- **Threshold rule**: flag any pair with cosine similarity < 0.40 as suspicious.
- **Cleanlab**: treat similarity scores below the median as noisy labels and use
  `find_label_issues` to rank the most suspicious samples.

### Step 4 — Evaluate and visualise

The script prints precision, recall, and F1 for both detectors, shows the top 10 flagged pairs,
and saves a UMAP scatter plot (`sleeper_agent_umap.png`) coloured by clean / poisoned status.

---

## Expected Outputs

```
[INFO] Dataset: 3000 pairs (2940 clean, 60 poisoned)
[DETECTION] Threshold detector  — Precision: 0.94  Recall: 0.97  F1: 0.95
[DETECTION] Cleanlab detector   — Precision: 0.88  Recall: 0.92  F1: 0.90
[TOP 10 FLAGGED PAIRS] ...
UMAP saved → sleeper_agent_umap.png
```

Exact numbers will vary with random seeds but precision and recall should both exceed 0.85.

---

## Discussion Questions

1. Why does semantic mismatch between instruction and response serve as a reliable poisoning signal?
2. What kinds of sleeper agent payloads would be hardest to detect with cosine similarity alone?
3. How would you extend this pipeline to monitor live fine-tuning jobs in a CI/CD workflow?
4. What is the trade-off between a low detection threshold (high recall) and a high one (high precision) in a medical context?
