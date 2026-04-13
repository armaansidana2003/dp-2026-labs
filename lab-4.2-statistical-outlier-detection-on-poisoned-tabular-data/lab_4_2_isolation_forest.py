"""
Lab 4.2 — Isolation Forest Anomaly Detection for Data Poisoning
================================================================
Data Poisoning Protection Course

Generates a synthetic tabular dataset (UCI Adult-style), injects 500 crafted
adversarial records, then compares Z-score and Isolation Forest detectors
across multiple contamination thresholds.

Run:
    python lab_4_2_isolation_forest.py

Outputs:
    isolation_forest_results.png  — Precision/Recall/F1 vs contamination rate
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Generate synthetic UCI Adult-style dataset
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TASK 1 — Generate Synthetic Dataset")
print("="*60)

N_CLEAN   = 1000
N_POISON  = 500

# ── Clean records ─────────────────────────────────────────────────────────────
# Age: bimodal around working-age population (25–55)
age_clean = np.clip(
    rng.integers(17, 90, size=N_CLEAN).astype(float)
    + rng.normal(0, 3, N_CLEAN),
    17, 90
).astype(int)

# Hours per week: roughly normal around 40
hours_clean = np.clip(
    rng.normal(40, 12, N_CLEAN),
    1, 80
).astype(int)

# Education number: 1–16 (1=preschool, 16=doctorate)
# Weighted toward high-school/some-college range
edu_probs   = np.array([1,1,1,1,2,2,5,10,8,5,4,3,2,1,1,1], dtype=float)
edu_probs  /= edu_probs.sum()
edu_clean   = rng.choice(np.arange(1, 17), size=N_CLEAN, p=edu_probs)

# Income: logistic model — older + more education + more hours → higher income
log_odds_clean = (
    0.05 * (age_clean - 38)
    + 0.15 * (edu_clean - 9)
    + 0.03 * (hours_clean - 40)
    + rng.normal(0, 0.5, N_CLEAN)
)
prob_high_clean = 1 / (1 + np.exp(-log_odds_clean))
income_clean    = (prob_high_clean > 0.5).astype(int)

# ── Adversarial (poisoned) records ────────────────────────────────────────────
# Crafted to have extreme feature values but the WRONG income label.
# High education (16) + high hours (95–99) + young age (18–25)
# → should logically be income=1, but we label them income=0 (wrong label).
# This creates a cluster of high-feature, low-income records that are
# statistically anomalous and should poison a downstream classifier.

age_poison   = rng.integers(18, 26, size=N_POISON)
hours_poison = rng.integers(95, 100, size=N_POISON)
edu_poison   = np.full(N_POISON, 16, dtype=int)      # all at max education
income_poison = np.zeros(N_POISON, dtype=int)        # wrong label (should be 1)

# ── Combine into a single DataFrame ───────────────────────────────────────────
df_clean = pd.DataFrame({
    "age":           age_clean,
    "hours_per_week": hours_clean,
    "education_num": edu_clean,
    "income":        income_clean,
    "is_poisoned":   0,
})

df_poison = pd.DataFrame({
    "age":           age_poison,
    "hours_per_week": hours_poison,
    "education_num": edu_poison,
    "income":        income_poison,
    "is_poisoned":   1,
})

df = pd.concat([df_clean, df_poison], ignore_index=True)
# Shuffle so poisoned records aren't all at the end
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Ground-truth label (1 = poisoned, 0 = clean)
y_true = df["is_poisoned"].values

FEATURE_COLS = ["age", "hours_per_week", "education_num"]
X = df[FEATURE_COLS].values.astype(float)

print(f"  Total records : {len(df)}")
print(f"  Clean         : {(y_true == 0).sum()}")
print(f"  Poisoned      : {(y_true == 1).sum()}")
print(f"\n  Feature statistics (all records):")
print(df[FEATURE_COLS].describe().round(2).to_string())
print(f"\n  Income distribution (clean)  : {dict(df_clean['income'].value_counts().sort_index())}")
print(f"  Income distribution (poison) : {dict(df_poison['income'].value_counts().sort_index())}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Z-score baseline detection
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TASK 2 — Z-Score Baseline Detection (|z| > 3)")
print("="*60)

# Standardize features
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Flag any record where at least one feature has |z| > 3
z_scores    = np.abs(X_scaled)
zscore_flag = (z_scores > 3).any(axis=1).astype(int)  # 1 = anomalous

n_zscore_flagged = zscore_flag.sum()

# Suppress UndefinedMetricWarning when precision/recall are 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    zscore_prec = precision_score(y_true, zscore_flag, zero_division=0)
    zscore_rec  = recall_score(y_true, zscore_flag, zero_division=0)
    zscore_f1   = f1_score(y_true, zscore_flag, zero_division=0)

print(f"  Records flagged : {n_zscore_flagged}")
print(f"  Precision       : {zscore_prec:.4f}")
print(f"  Recall          : {zscore_rec:.4f}")
print(f"  F1              : {zscore_f1:.4f}")

# Per-feature breakdown of how many flagged records triggered which feature
for col_i, col_name in enumerate(FEATURE_COLS):
    n_triggered = (z_scores[:, col_i] > 3).sum()
    print(f"    |z| > 3 on '{col_name}': {n_triggered} records")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Isolation Forest at multiple contamination rates
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TASK 3 — Isolation Forest Contamination Sweep")
print("="*60)

CONTAMINATION_RATES = [0.005, 0.01, 0.02, 0.05]

results = []   # list of dicts for the comparison table and plot

for rate in CONTAMINATION_RATES:
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=rate,
        max_samples="auto",
        random_state=SEED,
        n_jobs=-1,
    )
    # fit_predict returns +1 (inlier) or -1 (outlier/anomaly)
    raw_pred  = iso_forest.fit_predict(X_scaled)
    iso_flag  = (raw_pred == -1).astype(int)   # 1 = flagged as anomaly

    n_flagged = iso_flag.sum()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prec = precision_score(y_true, iso_flag, zero_division=0)
        rec  = recall_score(y_true, iso_flag, zero_division=0)
        f1   = f1_score(y_true, iso_flag, zero_division=0)

    results.append({
        "contamination": rate,
        "n_flagged":     n_flagged,
        "precision":     prec,
        "recall":        rec,
        "f1":            f1,
    })

# ── Print comparison table ────────────────────────────────────────────────────
print(f"\n  {'Contamination':>13} | {'Flagged':>7} | {'Precision':>9} | {'Recall':>8} | {'F1':>8}")
print("  " + "-"*57)

# First row: Z-score baseline for comparison
print(f"  {'Z-score (>3σ)':>13} | {n_zscore_flagged:>7} | {zscore_prec:>9.4f} | {zscore_rec:>8.4f} | {zscore_f1:>8.4f}")
print("  " + "-"*57)

for row in results:
    print(f"  {row['contamination']:>13.4f} | {row['n_flagged']:>7} | "
          f"{row['precision']:>9.4f} | {row['recall']:>8.4f} | {row['f1']:>8.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualization
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TASK 4 — Generate Visualization")
print("="*60)

rates      = [r["contamination"] for r in results]
precisions = [r["precision"]     for r in results]
recalls    = [r["recall"]        for r in results]
f1s        = [r["f1"]            for r in results]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Left panel: P/R/F1 vs contamination rate ──────────────────────────────────
ax = axes[0]
ax.plot(rates, precisions, "o-", color="steelblue",  linewidth=2, label="Precision")
ax.plot(rates, recalls,    "s-", color="tomato",     linewidth=2, label="Recall")
ax.plot(rates, f1s,        "^-", color="seagreen",   linewidth=2, label="F1")

# Baseline lines (horizontal)
ax.axhline(zscore_prec, color="steelblue",  linestyle="--", alpha=0.5,
           label=f"Z-score Precision ({zscore_prec:.2f})")
ax.axhline(zscore_rec,  color="tomato",     linestyle="--", alpha=0.5,
           label=f"Z-score Recall ({zscore_rec:.2f})")
ax.axhline(zscore_f1,   color="seagreen",   linestyle="--", alpha=0.5,
           label=f"Z-score F1 ({zscore_f1:.2f})")

ax.set_xlabel("Isolation Forest Contamination Rate", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Isolation Forest: Precision / Recall / F1\nvs Contamination Rate", fontsize=13)
ax.set_ylim(0, 1.05)
ax.set_xscale("log")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)

# Annotate each point with its F1 value
for rate, f1 in zip(rates, f1s):
    ax.annotate(f"F1={f1:.2f}", xy=(rate, f1),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color="seagreen")

# ── Right panel: Feature scatter of clean vs poisoned ─────────────────────────
ax2 = axes[1]
clean_mask  = y_true == 0
poison_mask = y_true == 1

ax2.scatter(
    df.loc[clean_mask,  "hours_per_week"],
    df.loc[clean_mask,  "education_num"],
    s=6, alpha=0.3, color="steelblue", label="Clean"
)
ax2.scatter(
    df.loc[poison_mask, "hours_per_week"],
    df.loc[poison_mask, "education_num"],
    s=10, alpha=0.6, color="tomato", label="Poisoned", marker="X"
)
ax2.set_xlabel("Hours per Week", fontsize=12)
ax2.set_ylabel("Education Number", fontsize=12)
ax2.set_title("Feature Space: Clean vs Poisoned Records\n"
              "(hours_per_week × education_num)", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle("Lab 4.2 — Isolation Forest Data Poisoning Detection", fontsize=14, y=1.01)
plt.tight_layout()

output_path = "isolation_forest_results.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {output_path}")


# ── Final summary ─────────────────────────────────────────────────────────────
best_f1_row   = max(results, key=lambda r: r["f1"])
best_rec_row  = max(results, key=lambda r: r["recall"])

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Dataset          : {N_CLEAN} clean + {N_POISON} poisoned = {len(df)} total")
print(f"  Attack type      : High-feature wrong-label injection")
print(f"  Z-score baseline : Precision={zscore_prec:.3f}  Recall={zscore_rec:.3f}  F1={zscore_f1:.3f}")
print(f"  Best IF F1       : {best_f1_row['f1']:.3f} at contamination={best_f1_row['contamination']}")
print(f"  Best IF Recall   : {best_rec_row['recall']:.3f} at contamination={best_rec_row['contamination']}")
print(f"  Plot             : {output_path}")
print("="*60)
print("\nLab 4.2 complete.")
