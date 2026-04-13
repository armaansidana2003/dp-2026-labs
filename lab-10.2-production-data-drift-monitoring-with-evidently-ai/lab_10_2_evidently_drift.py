"""
Lab 10.2 — Production Drift Monitoring with Evidently AI
=========================================================
Simulates 5 inference batches of increasing drift severity against a
reference fraud-detection dataset and monitors for adversarial distribution
shift using Evidently AI.

Run:
    python lab_10_2_evidently_drift.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Constants and configuration
# ---------------------------------------------------------------------------

ALERT_THRESHOLD     = 0.30   # drift_share above this is suspicious
ALERT_CONSECUTIVE   = 2      # trigger alert if N consecutive batches exceed threshold
ROLLBACK_THRESHOLD  = 0.50   # drift_share above this is severe
ROLLBACK_CONSECUTIVE = 3     # recommend rollback if N consecutive batches exceed

NUMERICAL_FEATURES  = ["age", "income", "transaction_amount"]
CAT_FEATURES        = ["merchant_category"]
ALL_FEATURES        = NUMERICAL_FEATURES + CAT_FEATURES
N_FEATURES          = len(ALL_FEATURES)

MERCHANT_CATEGORIES = ["grocery", "electronics", "travel", "restaurant", "online"]

# ---------------------------------------------------------------------------
# 2. Data generation
# ---------------------------------------------------------------------------

def generate_reference(n: int = 5000) -> pd.DataFrame:
    """Generate clean reference fraud detection dataset."""
    np.random.seed(SEED)
    return pd.DataFrame({
        "age":                np.random.randint(18, 75, size=n),
        "income":             np.random.normal(70_000, 20_000, size=n).clip(20_000, 200_000),
        "transaction_amount": np.random.exponential(scale=150, size=n).clip(1, 5_000),
        "merchant_category":  np.random.choice(MERCHANT_CATEGORIES, size=n),
        "label":              (np.random.rand(n) < 0.08).astype(int),
    })


def generate_batch(reference: pd.DataFrame, drift_level: float,
                   n: int = 1000, batch_seed: int = 0) -> pd.DataFrame:
    """
    Generate an inference batch with increasing drift.

    drift_level: 0.0 = no drift, 1.0 = severe shift on all features.
    Each feature is shifted proportionally to drift_level.
    """
    np.random.seed(SEED + batch_seed)

    # --- Age: shift mean upward as drift increases ---
    age_mean = 46 + drift_level * 25       # reference mean ≈ 46
    age = np.random.normal(age_mean, 10, size=n).clip(18, 90).astype(int)

    # --- Income: shift distribution lower (simulate economic stress) ---
    income_mean = 70_000 - drift_level * 45_000
    income = np.random.normal(income_mean, 15_000, size=n).clip(5_000, 200_000)

    # --- Transaction amount: inflate (simulate adversarial large transactions) ---
    txn_scale = 150 + drift_level * 1_000
    txn = np.random.exponential(scale=txn_scale, size=n).clip(1, 10_000)

    # --- Merchant category: shift toward electronics/online ---
    if drift_level < 0.3:
        cat_probs = [0.25, 0.20, 0.20, 0.20, 0.15]
    elif drift_level < 0.6:
        cat_probs = [0.10, 0.35, 0.10, 0.10, 0.35]
    else:
        cat_probs = [0.05, 0.45, 0.05, 0.05, 0.40]

    categories = np.random.choice(MERCHANT_CATEGORIES, size=n, p=cat_probs)

    return pd.DataFrame({
        "age":                age,
        "income":             income.round(2),
        "transaction_amount": txn.round(2),
        "merchant_category":  categories,
        "label":              (np.random.rand(n) < 0.08 + drift_level * 0.2).astype(int),
    })

# ---------------------------------------------------------------------------
# 3. Evidently drift analysis
# ---------------------------------------------------------------------------

def run_evidently_report(reference: pd.DataFrame,
                          current: pd.DataFrame,
                          batch_name: str) -> dict:
    """
    Run Evidently DataDriftPreset.
    Returns dict with: drift_share, drifted_features, feature_scores.
    Falls back to manual KS/chi2 tests if Evidently is unavailable.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference[ALL_FEATURES],
                   current_data=current[ALL_FEATURES])

        result = report.as_dict()
        metrics = result["metrics"]

        # Navigate Evidently v0.4+ JSON structure
        dataset_drift = None
        feature_drift = {}

        for metric in metrics:
            mid = metric.get("metric", "")
            res = metric.get("result", {})

            if "DatasetDriftMetric" in mid or "dataset_drift" in str(res):
                dataset_drift = res.get("drift_share", res.get("share_of_drifted_columns", None))

            if "ColumnDriftMetric" in mid or "column_name" in res:
                col   = res.get("column_name", "unknown")
                score = res.get("drift_score", res.get("stattest_threshold", 0.0))
                drifted = res.get("drift_detected", False)
                feature_drift[col] = {"score": score, "drifted": drifted}

        # If we couldn't parse the nested structure, try a simpler path
        if dataset_drift is None:
            for metric in metrics:
                res = metric.get("result", {})
                if "drift_share" in res:
                    dataset_drift = res["drift_share"]
                    break
                if "share_of_drifted_columns" in res:
                    dataset_drift = res["share_of_drifted_columns"]
                    break

        if dataset_drift is None:
            dataset_drift = len([f for f in feature_drift.values()
                                  if f["drifted"]]) / max(N_FEATURES, 1)

        return {
            "drift_share":       float(dataset_drift),
            "feature_drift":     feature_drift,
            "evidently_report":  report,
        }

    except ImportError:
        print("[WARN] evidently not installed — using manual KS/chi2 fallback.")
        return manual_drift_check(reference, current)
    except Exception as exc:
        print(f"[WARN] Evidently error ({exc}) — using manual fallback.")
        return manual_drift_check(reference, current)


def manual_drift_check(reference: pd.DataFrame,
                        current: pd.DataFrame) -> dict:
    """
    Manual drift detection using KS test for numerical features and
    chi-squared test for categorical features.
    Drift detected if p-value < 0.05.
    """
    from scipy import stats

    feature_drift = {}
    drifted_count = 0

    for col in NUMERICAL_FEATURES:
        stat, p_val = stats.ks_2samp(reference[col], current[col])
        drifted = p_val < 0.05
        if drifted:
            drifted_count += 1
        feature_drift[col] = {"score": float(stat), "p_value": float(p_val),
                               "drifted": drifted, "test": "KS"}

    for col in CAT_FEATURES:
        ref_counts = reference[col].value_counts()
        cur_counts = current[col].value_counts()
        all_cats   = set(ref_counts.index) | set(cur_counts.index)
        ref_freq   = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
        cur_freq   = np.array([cur_counts.get(c, 0) for c in all_cats], dtype=float)
        # Avoid zero expected frequencies
        ref_freq = ref_freq + 1e-9
        cur_freq = cur_freq + 1e-9
        chi2, p_val = stats.chisquare(f_obs=cur_freq / cur_freq.sum(),
                                       f_exp=ref_freq / ref_freq.sum())
        drifted = p_val < 0.05
        if drifted:
            drifted_count += 1
        feature_drift[col] = {"score": float(chi2), "p_value": float(p_val),
                               "drifted": drifted, "test": "chi2"}

    drift_share = drifted_count / N_FEATURES
    return {
        "drift_share":      drift_share,
        "feature_drift":    feature_drift,
        "evidently_report": None,
    }

# ---------------------------------------------------------------------------
# 4. Alert and rollback logic
# ---------------------------------------------------------------------------

def check_alerts(drift_history: list) -> tuple:
    """
    Returns (alert: bool, rollback: bool) based on consecutive batch history.
    drift_history: list of drift_share floats (most recent last).
    """
    alert    = False
    rollback = False

    # Alert: N consecutive batches above alert threshold
    if len(drift_history) >= ALERT_CONSECUTIVE:
        recent = drift_history[-ALERT_CONSECUTIVE:]
        if all(d > ALERT_THRESHOLD for d in recent):
            alert = True

    # Rollback: N consecutive batches above rollback threshold
    if len(drift_history) >= ROLLBACK_CONSECUTIVE:
        recent = drift_history[-ROLLBACK_CONSECUTIVE:]
        if all(d > ROLLBACK_THRESHOLD for d in recent):
            rollback = True

    return alert, rollback

# ---------------------------------------------------------------------------
# 5. Print drift table
# ---------------------------------------------------------------------------

def print_batch_summary(batch_num: int, drift_share: float,
                          feature_drift: dict) -> None:
    drifted = sum(1 for v in feature_drift.values() if v.get("drifted", False))
    total   = len(feature_drift)
    print(f"\nBatch {batch_num} | drift_share: {drift_share:.2f} | "
          f"Features drifted: {drifted}/{total}")
    print(f"  {'Feature':<25s} {'Score':>8}  {'Drifted':>8}")
    print("  " + "-" * 45)
    for feat, info in feature_drift.items():
        score   = info.get("score", 0.0)
        drifted = "YES" if info.get("drifted", False) else "no"
        print(f"  {feat:<25s} {score:>8.4f}  {drifted:>8}")

# ---------------------------------------------------------------------------
# 6. Plot drift over time
# ---------------------------------------------------------------------------

def plot_drift_trend(drift_shares: list,
                      out_path: str = "drift_over_time.png") -> None:
    batches = list(range(1, len(drift_shares) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(batches, drift_shares, marker="o", linewidth=2,
            color="steelblue", label="Drift Share")
    ax.axhline(ALERT_THRESHOLD, color="orange", linestyle="--",
               label=f"Alert threshold ({ALERT_THRESHOLD})")
    ax.axhline(ROLLBACK_THRESHOLD, color="crimson", linestyle="--",
               label=f"Rollback threshold ({ROLLBACK_THRESHOLD})")

    # Shade alert zone
    ax.fill_between(batches, ALERT_THRESHOLD, [max(d, ALERT_THRESHOLD) for d in drift_shares],
                    alpha=0.15, color="orange")
    ax.fill_between(batches, ROLLBACK_THRESHOLD,
                    [max(d, ROLLBACK_THRESHOLD) for d in drift_shares],
                    alpha=0.25, color="crimson")

    ax.set_xlabel("Inference Batch")
    ax.set_ylabel("Drift Share (fraction of features drifted)")
    ax.set_title("Adversarial Drift Monitoring — Drift Share Over Time\n"
                 "Simulated attack: gradual feature distribution shift",
                 fontsize=12)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xticks(batches)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n[INFO] Drift trend plot saved → {out_path}")

# ---------------------------------------------------------------------------
# 7. Save Evidently HTML report for batch 4
# ---------------------------------------------------------------------------

def save_html_report(evidently_report, out_path: str = "drift_monitoring_report.html") -> None:
    if evidently_report is None:
        print("[WARN] No Evidently report object — HTML report skipped.")
        return
    try:
        evidently_report.save_html(out_path)
        print(f"[INFO] Evidently HTML report saved → {out_path}")
    except Exception as exc:
        print(f"[WARN] Could not save HTML report: {exc}")

# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Lab 10.2 — Production Drift Monitoring with Evidently AI")
    print("=" * 60)

    # ---- Generate reference dataset ----
    print("\n[INFO] Generating reference dataset (5 000 rows) …")
    reference = generate_reference(n=5000)
    print(f"       Shape: {reference.shape}  |  "
          f"Fraud rate: {reference['label'].mean():.1%}")

    # Batch configurations: (drift_level, description)
    batch_configs = [
        (0.00, "no drift — clean baseline"),
        (0.10, "10% drift — slight age shift"),
        (0.30, "30% drift — multiple features shifting"),
        (0.50, "50% drift — significant adversarial traffic"),
        (0.70, "70% drift — severe poisoning attack"),
    ]

    drift_shares   = []
    html_batch4_report = None

    for batch_num, (drift_level, description) in enumerate(batch_configs, 1):
        print(f"\n{'─'*60}")
        print(f"[BATCH {batch_num}] drift_level={drift_level:.0%}  — {description}")

        current = generate_batch(reference, drift_level,
                                  n=1000, batch_seed=batch_num)

        result = run_evidently_report(reference, current,
                                       batch_name=f"batch_{batch_num}")

        drift_share  = result["drift_share"]
        feature_drift = result["feature_drift"]
        drift_shares.append(drift_share)

        print_batch_summary(batch_num, drift_share, feature_drift)

        # Save Evidently HTML report for batch 4 (50% drift = most interesting)
        if batch_num == 4 and result.get("evidently_report") is not None:
            html_batch4_report = result["evidently_report"]

        # Alert / rollback logic
        alert, rollback = check_alerts(drift_shares)

        if rollback:
            print(f"\n  [ROLLBACK] Severe drift sustained for {ROLLBACK_CONSECUTIVE}+ "
                  "consecutive batches.")
            print("             Recommend immediate model rollback to last clean checkpoint.")
        elif alert:
            print(f"\n  [ALERT]  Possible adversarial drift detected — "
                  f"{ALERT_CONSECUTIVE}+ consecutive batches above "
                  f"{ALERT_THRESHOLD} threshold.")
            print("           Investigate traffic source and retrain on clean data.")

    # ---- Save HTML report ----
    save_html_report(html_batch4_report, "drift_monitoring_report.html")

    # ---- Plot drift trend ----
    plot_drift_trend(drift_shares, "drift_over_time.png")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("DRIFT MONITORING SUMMARY")
    print("=" * 60)
    print(f"  {'Batch':<8} {'Drift Share':>12}  {'Status'}")
    print("  " + "-" * 40)
    for i, ds in enumerate(drift_shares, 1):
        status = ("SEVERE" if ds > ROLLBACK_THRESHOLD
                  else "ALERT" if ds > ALERT_THRESHOLD
                  else "OK")
        print(f"  {i:<8} {ds:>12.2f}  {status}")

    print("\n[DONE] Lab 10.2 complete.")


if __name__ == "__main__":
    main()
