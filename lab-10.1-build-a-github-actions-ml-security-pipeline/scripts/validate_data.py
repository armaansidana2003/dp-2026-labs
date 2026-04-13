"""
scripts/validate_data.py
========================
Runs a Great Expectations validation suite against data/train.csv.
Exits with code 1 if any expectation fails (CI gate).
"""

import sys
import json
import pathlib
import pandas as pd

# ---------------------------------------------------------------------------
# Great Expectations is optional — fall back to manual checks if not installed
# ---------------------------------------------------------------------------
try:
    import great_expectations as gx
    HAS_GX = True
except ImportError:
    HAS_GX = False
    print("[WARN] great-expectations not installed — running manual checks only.")

DATA_PATH   = pathlib.Path("data/train.csv")
RESULT_PATH = pathlib.Path("ge_validation_result.json")

# ---------------------------------------------------------------------------
# Manual / fallback validation logic
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS  = ["age", "income", "transaction_amount", "merchant_category", "label"]
ALLOWED_LABELS    = {0, 1}
NON_NULL_COLUMNS  = ["age", "income", "transaction_amount", "label"]

def manual_validate(df: pd.DataFrame) -> list:
    """Return list of (check_name, passed, message) tuples."""
    results = []

    # 1. Required columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    results.append((
        "required_columns_present",
        len(missing) == 0,
        f"Missing columns: {missing}" if missing else "All required columns present.",
    ))

    # 2. No null values in critical columns
    for col in NON_NULL_COLUMNS:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        results.append((
            f"no_nulls_in_{col}",
            null_count == 0,
            f"{null_count} nulls found in '{col}'." if null_count else f"'{col}' has no nulls.",
        ))

    # 3. Label values restricted to {0, 1}
    if "label" in df.columns:
        bad_labels = set(df["label"].unique()) - ALLOWED_LABELS
        results.append((
            "label_values_valid",
            len(bad_labels) == 0,
            f"Unexpected label values: {bad_labels}" if bad_labels else "Labels are valid {0, 1}.",
        ))

    # 4. Age within plausible range [0, 120]
    if "age" in df.columns:
        out_of_range = ((df["age"] < 0) | (df["age"] > 120)).sum()
        results.append((
            "age_in_range",
            out_of_range == 0,
            f"{out_of_range} rows with age out of [0, 120]." if out_of_range else "Age range valid.",
        ))

    # 5. Row count at least 100
    results.append((
        "minimum_row_count",
        len(df) >= 100,
        f"Only {len(df)} rows — expected >= 100." if len(df) < 100 else f"{len(df)} rows — OK.",
    ))

    return results


# ---------------------------------------------------------------------------
# Great Expectations validation (when available)
# ---------------------------------------------------------------------------

def gx_validate(df: pd.DataFrame) -> list:
    """
    Run a minimal GX suite and return (check_name, passed, message) tuples.
    Uses the fluent / v1 API (GX >= 1.0).
    """
    try:
        context = gx.get_context(mode="ephemeral")
        data_source = context.sources.add_pandas("lab_source")
        asset = data_source.add_dataframe_asset("train")
        batch_request = asset.build_batch_request(dataframe=df)
        suite = context.add_expectation_suite("lab_suite")

        expectations = [
            gx.expectations.ExpectColumnToExist(column="age"),
            gx.expectations.ExpectColumnToExist(column="label"),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="age", min_value=0, max_value=120
            ),
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="label", value_set=[0, 1]
            ),
            gx.expectations.ExpectTableRowCountToBeGreaterThan(value=99),
        ]
        for exp in expectations:
            suite.add_expectation(exp)

        validator  = context.get_validator(batch_request=batch_request,
                                           expectation_suite=suite)
        gx_results = validator.validate()

        results = []
        for res in gx_results.results:
            name = res.expectation_config.expectation_type
            passed = bool(res.success)
            msg    = "Passed." if passed else str(res.result)
            results.append((name, passed, msg))

        return results

    except Exception as exc:
        print(f"[WARN] GX validation raised an exception: {exc}")
        print("[WARN] Falling back to manual checks.")
        return manual_validate(df)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DATA_PATH.exists():
        print(f"[ERROR] {DATA_PATH} not found. Run data/generate_sample_data.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded {DATA_PATH}: {len(df)} rows, {len(df.columns)} columns.")

    results = gx_validate(df) if HAS_GX else manual_validate(df)

    all_passed = True
    report     = {"checks": []}

    for name, passed, msg in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        report["checks"].append({"name": name, "passed": passed, "message": msg})
        if not passed:
            all_passed = False

    report["overall"] = "PASS" if all_passed else "FAIL"
    RESULT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n[INFO] Validation report saved → {RESULT_PATH}")

    if not all_passed:
        print("[ERROR] Data validation FAILED — aborting pipeline.")
        sys.exit(1)

    print("[OK] All data validation checks passed.")


if __name__ == "__main__":
    main()
