# Lab 10.2 — Production Drift Monitoring with Evidently AI

## Overview

Distribution shift is one of the most insidious attack vectors against production ML systems.
An adversary who can influence live inference traffic can gradually poison the effective feature
distribution, degrading model performance or triggering targeted misbehaviour.

This lab uses Evidently AI to monitor five inference batches of increasing drift severity,
implements automated alerting logic, and produces an HTML report suitable for a security
post-mortem.

---

## Objectives

By the end of this lab you will be able to:

1. Generate reference and inference datasets that simulate adversarial drift.
2. Use Evidently's `DataDriftPreset` to compute per-feature drift scores.
3. Implement consecutive-batch alert logic that flags potential adversarial traffic.
4. Recommend rollback based on sustained severe drift.
5. Visualise drift share over time and save an Evidently HTML report.

---

## Prerequisites

- Python 3.10 or 3.11
- No ML deployment experience required — everything is simulated

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Instructions

### Step 1 — Run the script

```bash
python lab_10_2_evidently_drift.py
```

### Step 2 — Read the console output

The script prints a drift table for each batch showing drift share and per-feature scores.
Watch for the ALERT and ROLLBACK messages.

### Step 3 — Open the HTML report

```bash
open drift_monitoring_report.html   # macOS
xdg-open drift_monitoring_report.html  # Linux
start drift_monitoring_report.html  # Windows
```

### Step 4 — Inspect the drift trend plot

Open `drift_over_time.png` to see how drift share escalates across batches and where the
alert threshold is crossed.

---

## Expected Outputs

```
Batch 1 | drift_share: 0.00 | Features drifted: 0/4
Batch 2 | drift_share: 0.25 | Features drifted: 1/4
Batch 3 | drift_share: 0.50 | Features drifted: 2/4
Batch 4 | drift_share: 0.75 | Features drifted: 3/4
[ALERT]  Possible adversarial drift detected — 2+ consecutive batches above 0.30 threshold.
Batch 5 | drift_share: 1.00 | Features drifted: 4/4
[ROLLBACK] Severe drift sustained for 3+ batches. Recommend model rollback immediately.
drift_monitoring_report.html saved.
drift_over_time.png saved.
```

---

## Discussion Questions

1. What is the difference between natural covariate shift and adversarial input manipulation?
2. Why does monitoring drift share over *consecutive* batches reduce false positives?
3. At what drift share threshold would you trigger automatic model rollback in a medical context vs. a fraud detection context?
4. How would you extend this monitoring to detect data poisoning during model retraining (not just inference)?
