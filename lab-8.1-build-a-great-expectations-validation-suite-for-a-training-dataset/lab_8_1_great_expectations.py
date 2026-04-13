"""
Lab 8.1 — Great Expectations Data Quality Gates
================================================
Course : Data Poisoning Protection
Author : Armaan Sidana

This script demonstrates how to use Great Expectations to build a
validation suite that catches data-poisoning attacks before they
reach an ML training pipeline.

Run:
    python lab_8_1_great_expectations.py

Exit codes:
    0 — clean data passed, attacked data correctly flagged
    1 — validation produced unexpected results
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Great Expectations imports
# We use the "core" (non-Fluent) API so the script works offline without
# a DataContext directory.
# ---------------------------------------------------------------------------
try:
    import great_expectations as gx
    from great_expectations.core import ExpectationSuite, ExpectationConfiguration
    from great_expectations.dataset import PandasDataset
except ImportError as exc:
    print(f"[ERROR] Great Expectations not installed: {exc}")
    print("Run:  pip install -r requirements.txt")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_CLEAN = 2000
LABEL_FLIP_RATE = 0.08   # 8 % of clean records
N_INJECTED = 400          # extra crafted rows appended
CLEAN_CSV = "clean_data.csv"
ATTACKED_CSV = "attacked_data.csv"

# Acceptable ranges derived from domain knowledge (and verified on clean data)
AGE_MIN, AGE_MAX = 18, 80
INCOME_MIN, INCOME_MAX = 10_000, 200_000
TXN_MIN, TXN_MAX = 1, 10_000
ROW_MIN, ROW_MAX = 1_800, 2_200          # ±10 % tolerance on 2 000 rows
MEAN_AGE_MIN, MEAN_AGE_MAX = 40.0, 55.0
MEAN_INCOME_MIN, MEAN_INCOME_MAX = 80_000.0, 130_000.0
MEAN_TXN_MIN, MEAN_TXN_MAX = 3_000.0, 7_000.0


# ===========================================================================
# SECTION 1 — Dataset Generation
# ===========================================================================

def generate_clean_dataset(n: int = N_CLEAN, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic fraud-detection dataset.

    Columns
    -------
    age               : int,   18 – 80
    income            : float, $10 000 – $200 000
    transaction_amount: float, $1 – $10 000
    label             : int,   0 = legitimate, 1 = fraud
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(AGE_MIN, AGE_MAX + 1, size=n)
    income = rng.uniform(INCOME_MIN, INCOME_MAX, size=n)

    # Transaction amounts are right-skewed (most are small)
    transaction_amount = rng.exponential(scale=2_000, size=n).clip(TXN_MIN, TXN_MAX)

    # Fraud rate ~10 %
    label = (rng.random(size=n) < 0.10).astype(int)

    df = pd.DataFrame({
        "age": age.astype(int),
        "income": income.round(2),
        "transaction_amount": transaction_amount.round(2),
        "label": label,
    })

    print(f"[+] Generated clean dataset: {len(df)} rows")
    print(f"    age       : {df['age'].min()} – {df['age'].max()}")
    print(f"    income    : {df['income'].min():,.0f} – {df['income'].max():,.0f}")
    print(f"    txn_amount: {df['transaction_amount'].min():,.2f} – {df['transaction_amount'].max():,.2f}")
    print(f"    fraud rate: {df['label'].mean():.1%}")
    return df


def create_attacked_dataset(clean_df: pd.DataFrame, seed: int = RANDOM_SEED + 1) -> pd.DataFrame:
    """
    Build a poisoned dataset from clean_df by combining two attacks:

    Attack A — Label flip:
        Randomly flip LABEL_FLIP_RATE of the labels (0→1 or 1→0).

    Attack B — Out-of-range injection:
        Append N_INJECTED crafted rows with impossible feature values
        (age=999, income=-1, transaction_amount=-500) to evade simple
        range filters at the application layer while corrupting statistics.
    """
    rng = np.random.default_rng(seed)
    poisoned = clean_df.copy()

    # --- Attack A: label flip ---
    n_flip = int(len(poisoned) * LABEL_FLIP_RATE)
    flip_idx = rng.choice(len(poisoned), size=n_flip, replace=False)
    poisoned.loc[flip_idx, "label"] = 1 - poisoned.loc[flip_idx, "label"]
    print(f"[!] Attack A: flipped {n_flip} labels ({LABEL_FLIP_RATE:.0%} of {len(clean_df)})")

    # --- Attack B: out-of-range injection ---
    injected = pd.DataFrame({
        "age": [999] * N_INJECTED,           # impossible age
        "income": [-1.0] * N_INJECTED,       # negative income
        "transaction_amount": [-500.0] * N_INJECTED,  # impossible amount
        "label": [1] * N_INJECTED,           # all labeled fraud
    })
    poisoned = pd.concat([poisoned, injected], ignore_index=True)
    print(f"[!] Attack B: injected {N_INJECTED} crafted out-of-range rows")
    print(f"    Poisoned dataset size: {len(poisoned)} rows")
    return poisoned


# ===========================================================================
# SECTION 2 — Great Expectations Validation Suite
# ===========================================================================

def build_expectation_suite() -> ExpectationSuite:
    """
    Create an ExpectationSuite that encodes the statistical contract
    of a clean fraud-detection dataset.

    Each expectation corresponds to a property an adversary would
    violate if they inject crafted data or flip labels at scale.
    """
    suite = ExpectationSuite(expectation_suite_name="fraud_data_quality")

    # ------------------------------------------------------------------
    # 1. Range checks — out-of-range injection will violate these
    # ------------------------------------------------------------------
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "age", "min_value": AGE_MIN, "max_value": AGE_MAX},
        meta={"notes": "Human age must be 18–80 for this product"}
    ))
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "income", "min_value": INCOME_MIN, "max_value": INCOME_MAX},
        meta={"notes": "Income range validated against onboarding data"}
    ))
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "transaction_amount",
            "min_value": TXN_MIN,
            "max_value": TXN_MAX
        },
        meta={"notes": "Card limit is $10 000"}
    ))

    # ------------------------------------------------------------------
    # 2. Null checks — missing values indicate upstream corruption
    # ------------------------------------------------------------------
    for col in ["age", "income", "transaction_amount", "label"]:
        suite.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": col},
        ))

    # ------------------------------------------------------------------
    # 3. Label domain check — only {0, 1} are valid fraud labels
    # ------------------------------------------------------------------
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_in_set",
        kwargs={"column": "label", "value_set": [0, 1]},
        meta={"notes": "Binary fraud label"}
    ))

    # ------------------------------------------------------------------
    # 4. Row count — sudden size changes indicate injection or deletion
    # ------------------------------------------------------------------
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_table_row_count_to_be_between",
        kwargs={"min_value": ROW_MIN, "max_value": ROW_MAX},
        meta={"notes": "Daily batch should be ~2 000 records ±10 %"}
    ))

    # ------------------------------------------------------------------
    # 5. Distribution checks — mass label flips shift the mean
    # ------------------------------------------------------------------
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_mean_to_be_between",
        kwargs={"column": "age", "min_value": MEAN_AGE_MIN, "max_value": MEAN_AGE_MAX},
        meta={"notes": "Mean age of customers is historically ~47"}
    ))
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_mean_to_be_between",
        kwargs={
            "column": "income",
            "min_value": MEAN_INCOME_MIN,
            "max_value": MEAN_INCOME_MAX
        },
        meta={"notes": "Mean income is historically ~$105 000"}
    ))
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_mean_to_be_between",
        kwargs={
            "column": "transaction_amount",
            "min_value": MEAN_TXN_MIN,
            "max_value": MEAN_TXN_MAX
        },
        meta={"notes": "Mean transaction historically ~$2 000 (exponential dist)"}
    ))

    return suite


# ===========================================================================
# SECTION 3 — Validation Runner
# ===========================================================================

# Human-readable labels for each expectation type
EXPECTATION_LABELS = {
    "expect_column_values_to_be_between": lambda k: (
        f"'{k['column']}' values between {k['min_value']} and {k['max_value']}"
    ),
    "expect_column_values_to_not_be_null": lambda k: (
        f"No nulls in '{k['column']}'"
    ),
    "expect_column_values_to_be_in_set": lambda k: (
        f"'{k['column']}' values in {set(k['value_set'])}"
    ),
    "expect_table_row_count_to_be_between": lambda k: (
        f"Row count between {k['min_value']} and {k['max_value']}"
    ),
    "expect_column_mean_to_be_between": lambda k: (
        f"Mean '{k['column']}' between {k['min_value']} and {k['max_value']}"
    ),
}


def label_for(result) -> str:
    """Return a human-readable description for an expectation result."""
    etype = result.expectation_config.expectation_type
    kwargs = result.expectation_config.kwargs
    labeler = EXPECTATION_LABELS.get(etype)
    if labeler:
        return labeler(kwargs)
    return etype


def validate_dataframe(df: pd.DataFrame, suite: ExpectationSuite, name: str) -> bool:
    """
    Validate a DataFrame against the expectation suite.

    Parameters
    ----------
    df    : DataFrame to validate
    suite : Great Expectations suite
    name  : display name for output

    Returns
    -------
    bool  : True if all expectations pass, False otherwise
    """
    print(f"\n{'='*60}")
    print(f" Validating {name}")
    print(f"{'='*60}")

    # Wrap the DataFrame as a Great Expectations PandasDataset
    ge_df = PandasDataset(df, expectation_suite=suite)

    # Run the full suite
    results = ge_df.validate(expectation_suite=suite, result_format="SUMMARY")

    all_passed = True
    for r in results.results:
        passed = r.success
        label = label_for(r)

        if passed:
            marker = "  \u2713"   # ✓
            detail = ""
        else:
            marker = "  \u2717"   # ✗
            all_passed = False
            # Extract failure detail where available
            res = r.result
            detail_parts = []
            if "unexpected_count" in res and res["unexpected_count"]:
                pct = res.get("unexpected_percent", 0)
                detail_parts.append(
                    f"{res['unexpected_count']} unexpected values "
                    f"({pct:.1f}%)"
                )
            if "observed_value" in res:
                detail_parts.append(f"observed={res['observed_value']}")
            detail = " — " + ", ".join(detail_parts) if detail_parts else ""

        print(f"{marker} {label}{detail}")

    print()
    if all_passed:
        print(f"[RESULT] {name} PASSED all expectations.")
    else:
        print(f"[RESULT] {name} FAILED validation. Pipeline blocked.")

    return all_passed


# ===========================================================================
# SECTION 4 — Main
# ===========================================================================

def main():
    print("=" * 60)
    print(" Lab 8.1 — Great Expectations Data Quality Gates")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Generate datasets
    # ------------------------------------------------------------------
    print("\n--- Step 1: Generating datasets ---")
    clean_df = generate_clean_dataset()
    clean_df.to_csv(CLEAN_CSV, index=False)
    print(f"[+] Saved → {CLEAN_CSV}")

    attacked_df = create_attacked_dataset(clean_df)
    attacked_df.to_csv(ATTACKED_CSV, index=False)
    print(f"[+] Saved → {ATTACKED_CSV}")

    # ------------------------------------------------------------------
    # Step 2: Build the expectation suite from clean data statistics
    # ------------------------------------------------------------------
    print("\n--- Step 2: Building Expectation Suite ---")
    suite = build_expectation_suite()
    n_exp = len(suite.expectations)
    print(f"[+] Suite created with {n_exp} expectations")

    # ------------------------------------------------------------------
    # Step 3: Validate clean data — expect all to pass
    # ------------------------------------------------------------------
    print("\n--- Step 3: Validating clean data ---")
    clean_passed = validate_dataframe(
        pd.read_csv(CLEAN_CSV), suite, CLEAN_CSV
    )

    # ------------------------------------------------------------------
    # Step 4: Validate attacked data — expect failures
    # ------------------------------------------------------------------
    print("\n--- Step 4: Validating attacked data ---")
    attacked_passed = validate_dataframe(
        pd.read_csv(ATTACKED_CSV), suite, ATTACKED_CSV
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" FINAL SUMMARY")
    print("=" * 60)
    print(f"  clean_data.csv   : {'PASS \u2713' if clean_passed else 'FAIL \u2717'}")
    print(f"  attacked_data.csv: {'PASS \u2713' if attacked_passed else 'FAIL \u2717 (expected)'}")

    # Security gate: exit 1 if clean data failed (unexpected) OR
    # if the attacked data slipped through the gate (undetected attack).
    if not clean_passed:
        print("\n[ERROR] Clean data failed validation — check expectation thresholds.")
        sys.exit(1)

    if attacked_passed:
        print("\n[WARNING] Attacked data passed all validations — gate may be too loose.")
        sys.exit(1)

    print("\n[OK] Gate correctly allowed clean data and blocked poisoned data.")
    print("     In production, replace sys.exit(1) with a pipeline abort signal.")
    sys.exit(0)


if __name__ == "__main__":
    main()
