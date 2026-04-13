# Lab 10.1 — GitHub Actions ML Security Pipeline

## Overview

This lab implements a three-stage automated security gate for ML systems using GitHub Actions.
Every push to `main` and every pull request triggers data validation, model scanning, and
supply-chain auditing before any artifact reaches production.

---

## Objectives

1. Implement automated data validation with Great Expectations as a CI gate.
2. Run ModelScan and a simplified Neural Cleanse check on model artifacts.
3. Audit Python dependencies for CVEs with pip-audit.
4. Verify model artifact integrity with HMAC signatures.
5. Understand how to wire Slack notifications to pipeline failures.

---

## Prerequisites

- A GitHub account with Actions enabled on your repository.
- Repository secrets configured (see Setup below).
- Python 3.11.

---

## Setup Instructions

### 1 — Fork / push this folder to a GitHub repository

The `.github/workflows/ml-security.yml` file must be at the repository root.

### 2 — Configure GitHub Secrets

Navigate to: `Settings → Secrets and variables → Actions → New repository secret`

| Secret name           | Value                                              |
|-----------------------|----------------------------------------------------|
| `SLACK_WEBHOOK_URL`   | Your Slack Incoming Webhook URL                    |
| `MODEL_HMAC_SECRET`   | A 32-byte hex string (see `make sign-model`)       |
| `DATA_MANIFEST_HASH`  | SHA-256 of `data/train.csv` after generation       |

### 3 — Generate sample data

```bash
python data/generate_sample_data.py
```

This creates `data/train.csv` and `data/manifest.json`.

### 4 — Sign the model artifact

```bash
make sign-model
```

This creates `model.pt` (dummy) and `model.sig` (HMAC signature).

---

## How to Test

Push any change to `main` or open a pull request.  Watch the Actions tab for:

- `data-validation` job — green if data passes Great Expectations suite.
- `model-validation` job — green if no malicious serialisation or backdoor found.
- `supply-chain` job — green if no HIGH/CRITICAL CVEs and signature is valid.

To simulate a failure, corrupt `data/train.csv` or add a `numpy==1.21.0` (known CVE) to
`requirements.txt`.

---

## Expected Outputs

On success:

```
data-validation   ✓  All 3 Great Expectations checkpoints passed.
model-validation  ✓  No malicious serialisation found. Neural Cleanse clean.
supply-chain      ✓  0 HIGH/CRITICAL CVEs. Signature verified.
```

On failure, a Slack message is sent to the configured channel:
> "[ML Security] Pipeline FAILED on main — job: supply-chain"

---

## Local Testing

```bash
make test-local   # runs validate_data + check_data_integrity
make scan         # runs scan_model + run_neural_cleanse
make audit        # runs pip-audit + check_audit_results
```
