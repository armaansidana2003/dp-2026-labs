# Lab 4.2 — Isolation Forest Anomaly Detection for Data Poisoning

## Overview

Adversarial data poisoning often injects records with unusual feature combinations designed to shift a model's decision boundary. This lab compares a naive **Z-score baseline** against **Isolation Forest** — an ensemble anomaly detector — across multiple contamination rates, using a synthetic tabular dataset modeled on the UCI Adult income dataset.

---

## Objectives

By the end of this lab you will be able to:

1. Simulate a tabular poisoning attack with crafted adversarial records.
2. Apply Z-score anomaly detection as a reproducible baseline.
3. Configure and evaluate Isolation Forest across a range of contamination hyperparameters.
4. Compute precision, recall, and F1 against injected ground truth.
5. Select an operational contamination threshold using the precision/recall tradeoff curve.

---

## Prerequisites

- Python 3.9+
- No GPU required
- Install dependencies: `pip install -r requirements.txt`

---

## Background

**Isolation Forest** isolates observations by randomly selecting a feature and a split value. Anomalies require fewer splits to isolate than normal points, yielding shorter average path lengths. The anomaly score is a normalized function of path length — lower score = more anomalous.

In a poisoning context, injected records often have atypical feature combinations (e.g., age=17, hours-per-week=99, education-num=16) that stand out in random partition trees, making Isolation Forest a natural first-pass detector.

---

## Dataset Schema

| Feature | Type | Normal Range |
|---------|------|-------------|
| `age` | int | 17–90 |
| `hours_per_week` | int | 1–99 |
| `education_num` | int | 1–16 |
| `income` | int | 0 (<=50K), 1 (>50K) |

Adversarial records have `education_num=16`, `hours_per_week=99`, `age` in 18–25, but `income=0` — the wrong label for high-credential, high-effort workers.

---

## Step-by-Step Tasks

### Task 1 — Generate Synthetic Dataset (lines ~35–75)

- 1000 records are generated to approximate UCI Adult demographics.
- Examine the feature distributions printed to console.
- Verify that the poisoned records are appended (rows 1000–1499).

### Task 2 — Z-Score Baseline Detection (lines ~77–110)

- Standardize all numeric features.
- Flag any record where at least one feature has |z| > 3.
- Print precision, recall, F1 versus the injected ground truth mask.

### Task 3 — Isolation Forest Sweep (lines ~112–165)

- Run `IsolationForest` with contamination in `[0.005, 0.01, 0.02, 0.05]`.
- For each rate, collect predictions, compute P/R/F1, and print a formatted table.

### Task 4 — Visualization (lines ~167–210)

- Plot Precision, Recall, and F1 as a function of contamination rate.
- Save as `isolation_forest_results.png`.
- Identify the elbow point where precision starts falling sharply.

---

## Expected Outputs

```
=== Synthetic Dataset ===
Total records: 1500  |  Poisoned: 500  |  Clean: 1000

=== Z-Score Baseline (|z| > 3) ===
Flagged: 87  |  Precision: 0.91  |  Recall: 0.16  |  F1: 0.27

=== Isolation Forest Results ===
Contamination | Flagged |  Precision |   Recall |       F1
-------------------------------------------------------------
       0.0050 |       8 |     0.8750 |   0.0140 |   0.0275
       0.0100 |      15 |     0.8667 |   0.0260 |   0.0504
       0.0200 |      30 |     0.8333 |   0.0500 |   0.0943
       0.0500 |      75 |     0.8000 |   0.1200 |   0.2087

Saved: isolation_forest_results.png
```

(Exact numbers vary with random seed.)

---

## Discussion Questions

1. Z-score detection had high precision but very low recall. Why does it miss most poisoned records even though the adversarial features were designed to be extreme?
2. As the Isolation Forest contamination rate increases, recall rises but precision drops. What is the operational cost of each type of error in a real ML pipeline?
3. Our adversarial records used simple extremes. How would a sophisticated attacker craft records that **evade** Isolation Forest while still causing label poisoning?
4. Isolation Forest is an unsupervised method — it has no knowledge of the label column. What additional signal might a supervised poisoning detector exploit?
5. At what contamination rate would you deploy this detector in a production pipeline, and why?

---

## File Outputs

| File | Description |
|------|-------------|
| `isolation_forest_results.png` | Precision/Recall/F1 vs contamination rate |

---

## Grading Checklist

- [ ] Script runs end-to-end without errors
- [ ] Dataset generation prints correct counts (1000 clean + 500 poisoned)
- [ ] Z-score baseline table printed with correct columns
- [ ] Isolation Forest table shows 4 rows with correct metrics
- [ ] Plot saved with labeled axes and legend
- [ ] Discussion questions answered in lab report
