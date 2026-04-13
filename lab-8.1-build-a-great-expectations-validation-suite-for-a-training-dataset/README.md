# Lab 8.1 — Great Expectations Data Quality Gates

## Objectives

By the end of this lab you will be able to:

1. Build a Great Expectations validation suite that encodes the statistical properties of a clean training dataset.
2. Automatically detect data-poisoning attacks (label flips, out-of-range injections) before they enter the ML pipeline.
3. Interpret validation results and use exit codes to gate downstream pipeline stages.
4. Understand the difference between schema validation and distribution-level validation.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 – 3.11 |
| pip | latest |
| Great Expectations | 0.18.x |
| pandas | 2.x |
| numpy | 1.26.x |
| scikit-learn | 1.4.x |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Instructions

### Step 1 — Understand the scenario

You are a data engineer for a fraud-detection team. Your pipeline ingests transaction data nightly. An adversary has been inserting crafted records and flipping labels in the incoming data feed. Your job is to set up a validation gate that automatically rejects poisoned batches.

### Step 2 — Run the lab script

```bash
python lab_8_1_great_expectations.py
```

The script will:

- Generate a clean synthetic fraud-detection dataset (`clean_data.csv`, 2 000 rows).
- Build a Great Expectations suite from that clean data.
- Generate an attacked dataset (`attacked_data.csv`) with label flips and out-of-range injections.
- Validate both files and print a per-expectation pass/fail summary.
- Exit with code `0` if the clean data passes and code `1` if the attacked data fails.

### Step 3 — Inspect the CSV files

```bash
# Spot the injected records
python -c "
import pandas as pd
df = pd.read_csv('attacked_data.csv')
print(df[df['age'] > 100].head())
print(df[df['income'] < 0].head())
"
```

### Step 4 — Modify and extend

Try the following modifications to deepen your understanding:

- Tighten the `transaction_amount` upper bound from 10 000 to 5 000 and re-run validation.
- Add a new expectation: `expect_column_quantile_values_to_be_between` for `income`.
- Change the label-flip rate from 8 % to 20 % and observe which expectations catch it first.

---

## Expected Outputs

```
=== Validating clean_data.csv ===
  ✓ age values between 18 and 80
  ✓ income values between 10000 and 200000
  ✓ transaction_amount values between 1 and 10000
  ✓ No nulls in [age, income, transaction_amount, label]
  ✓ label values in {0, 1}
  ✓ Row count between 1800 and 2200
  ✓ Mean age between 40 and 55
  ✓ Mean income between 80000 and 130000
Clean data PASSED all expectations.

=== Validating attacked_data.csv ===
  ✗ age values between 18 and 80            — 400 unexpected values
  ✗ income values between 10000 and 200000  — 400 unexpected values
  ✓ No nulls in [age, income, transaction_amount, label]
  ✗ label values in {0, 1}                  — 0 unexpected (labels valid)
  ✗ Row count between 1800 and 2200         — found 2400
  ✗ Mean age between 40 and 55             — found 214.3
Attacked data FAILED validation. Blocking pipeline.
```

*Exact numbers will vary because synthetic data uses a random seed.*

---

## Discussion Questions

1. Great Expectations checks statistical properties of the data, not the model itself. At what point in a real ML pipeline would you place this gate — before training, after training, or both?
2. An attacker who knows your validation thresholds can craft records that pass all checks while still shifting the decision boundary. How would you defend against this "threshold-aware" adversary?
3. The `expect_column_mean_to_be_between` expectation catches distribution shift. What are the limitations of mean-based checks? Name at least two attack patterns that would evade them.
4. In a production system, should a validation failure block the pipeline hard (exit 1) or quarantine the batch for human review? Discuss the operational trade-offs.
5. How would you version your expectation suite alongside your model versions in a CI/CD pipeline?
