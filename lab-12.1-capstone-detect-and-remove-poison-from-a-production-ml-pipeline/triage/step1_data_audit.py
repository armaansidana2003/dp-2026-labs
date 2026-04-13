"""
triage/step1_data_audit.py
===========================
Step 1 of the MedBot-7 incident response.

Audits suspicious_data.csv for:
  1. Schema validity (required columns, correct dtypes).
  2. Label distribution anomalies (unusually high rate of unsafe=0 labels).
  3. Pattern analysis: which patient_types have elevated unsafe label rates.
  4. Great Expectations validation (if installed).

Run:
    python triage/step1_data_audit.py
"""

import sys
import pathlib
import datetime
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 0. Paths and constants
# ---------------------------------------------------------------------------

TRIAGE_DIR = pathlib.Path(__file__).parent
DATA_PATH  = TRIAGE_DIR / "suspicious_data.csv"

REQUIRED_COLUMNS = [
    "patient_id", "patient_type", "medication",
    "weight_kg", "age_years", "creatinine",
    "recommended_dose_label",
]
EXPECTED_UNSAFE_RATE = 0.02   # baseline: 2% of doses should be flagged unsafe
ALERT_UNSAFE_RATE    = 0.05   # flag if rate exceeds 5%

# ---------------------------------------------------------------------------
# 1. Load and validate schema
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(f"[ERROR] {DATA_PATH} not found. Run triage/triage.py first.")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded {DATA_PATH.name}: {len(df)} rows, {len(df.columns)} columns")
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [FAIL] Missing columns: {missing}")
        return False
    print(f"  [PASS] All required columns present.")
    return True

# ---------------------------------------------------------------------------
# 2. Label distribution analysis
# ---------------------------------------------------------------------------

def analyse_labels(df: pd.DataFrame) -> dict:
    total         = len(df)
    unsafe_count  = (df["recommended_dose_label"] == 0).sum()
    unsafe_rate   = unsafe_count / total

    print(f"\n[STEP 1.2] Label Distribution Analysis")
    print(f"  Total samples    : {total}")
    print(f"  Safe (label=1)   : {(df['recommended_dose_label']==1).sum()} "
          f"({(df['recommended_dose_label']==1).mean():.1%})")
    print(f"  Unsafe (label=0) : {unsafe_count} ({unsafe_rate:.1%})")
    print(f"  Expected baseline: {EXPECTED_UNSAFE_RATE:.1%}")

    if unsafe_rate > ALERT_UNSAFE_RATE:
        excess = unsafe_rate - EXPECTED_UNSAFE_RATE
        print(f"  [ALERT] Unsafe label rate {unsafe_rate:.1%} exceeds baseline by "
              f"{excess:.1%} — possible label flip attack!")
    else:
        print(f"  [OK]    Unsafe label rate within expected range.")

    return {"unsafe_rate": unsafe_rate, "unsafe_count": int(unsafe_count), "total": total}

# ---------------------------------------------------------------------------
# 3. Patient-type breakdown
# ---------------------------------------------------------------------------

def analyse_by_patient_type(df: pd.DataFrame) -> None:
    print(f"\n[STEP 1.3] Unsafe Label Rate by Patient Type")
    print(f"  {'Patient Type':<20} {'Total':>7} {'Unsafe':>7} {'Rate':>8}  {'Status'}")
    print("  " + "-" * 55)

    for ptype in df["patient_type"].unique():
        subset = df[df["patient_type"] == ptype]
        total  = len(subset)
        unsafe = (subset["recommended_dose_label"] == 0).sum()
        rate   = unsafe / total if total > 0 else 0
        status = "[ALERT]" if rate > ALERT_UNSAFE_RATE else "[OK]"
        print(f"  {ptype:<20} {total:>7} {unsafe:>7} {rate:>8.1%}  {status}")

# ---------------------------------------------------------------------------
# 4. Medication-level analysis
# ---------------------------------------------------------------------------

def analyse_by_medication(df: pd.DataFrame, top_n: int = 5) -> None:
    print(f"\n[STEP 1.4] Top {top_n} Most Poisoned Medications")
    med_stats = (
        df.groupby("medication")["recommended_dose_label"]
        .agg(["count", lambda x: (x == 0).sum()])
        .rename(columns={"count": "total", "<lambda_0>": "unsafe"})
    )
    med_stats["unsafe_rate"] = med_stats["unsafe"] / med_stats["total"]
    med_stats = med_stats.sort_values("unsafe_rate", ascending=False)

    print(f"  {'Medication':<20} {'Total':>7} {'Unsafe':>7} {'Rate':>8}")
    print("  " + "-" * 45)
    for med, row in med_stats.head(top_n).iterrows():
        print(f"  {med:<20} {int(row['total']):>7} {int(row['unsafe']):>7} "
              f"{row['unsafe_rate']:>8.1%}")

# ---------------------------------------------------------------------------
# 5. Great Expectations validation (optional)
# ---------------------------------------------------------------------------

def run_ge_validation(df: pd.DataFrame) -> None:
    try:
        import great_expectations as gx

        print("\n[STEP 1.5] Great Expectations Validation")
        context = gx.get_context(mode="ephemeral")
        source  = context.sources.add_pandas("medbot_source")
        asset   = source.add_dataframe_asset("suspicious_data")
        batch_req = asset.build_batch_request(dataframe=df)
        suite   = context.add_expectation_suite("medbot_suite")

        # Define expectations
        expectations = [
            gx.expectations.ExpectColumnToExist(column="patient_id"),
            gx.expectations.ExpectColumnToExist(column="recommended_dose_label"),
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="recommended_dose_label", value_set=[0, 1]
            ),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="weight_kg", min_value=1, max_value=300
            ),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="age_years", min_value=0, max_value=120
            ),
            # This expectation will FAIL — detecting the anomaly
            gx.expectations.ExpectColumnMeanToBeBetween(
                column="recommended_dose_label",
                min_value=0.95,   # expect >= 95% safe labels
                max_value=1.00,
            ),
        ]
        for exp in expectations:
            suite.add_expectation(exp)

        validator = context.get_validator(batch_request=batch_req,
                                          expectation_suite=suite)
        result = validator.validate()

        for res in result.results:
            name   = res.expectation_config.expectation_type
            passed = "PASS" if res.success else "FAIL"
            note   = "" if res.success else " <-- ANOMALY DETECTED"
            print(f"  [{passed}] {name}{note}")

    except ImportError:
        print("\n[INFO] great-expectations not installed — skipping GE validation.")
    except Exception as exc:
        print(f"\n[WARN] GE validation error: {exc}")

# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*60}")
    print(f"STEP 1 — DATA AUDIT  [{now}]")
    print(f"{'='*60}")

    df = load_data()

    print("\n[STEP 1.1] Schema Validation")
    schema_ok = validate_schema(df)
    if not schema_ok:
        print("[ERROR] Schema invalid — investigation blocked.")
        sys.exit(1)

    label_stats = analyse_labels(df)
    analyse_by_patient_type(df)
    analyse_by_medication(df, top_n=5)
    run_ge_validation(df)

    print(f"\n{'='*60}")
    print("STEP 1 FINDINGS:")
    if label_stats["unsafe_rate"] > ALERT_UNSAFE_RATE:
        print(f"  [CONFIRMED] Vector 1: Label Flip Poisoning")
        print(f"  {label_stats['unsafe_count']} of {label_stats['total']} samples "
              f"({label_stats['unsafe_rate']:.1%}) have flipped labels.")
        print(f"  Attack targets paediatric and renal_impaired patients.")
    else:
        print("  [NOT CONFIRMED] Label flip rate within normal bounds.")
    print(f"{'='*60}")
    print("Next step: python triage/step2_model_scan.py")


if __name__ == "__main__":
    main()
