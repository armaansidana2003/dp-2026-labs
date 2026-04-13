# Lab 8.2 — DVC Dataset Versioning & Integrity

## Objectives

By the end of this lab you will be able to:

1. Explain how DVC uses content-addressable storage and SHA-256 hashing to detect dataset tampering.
2. Implement a lightweight DVC-like versioning workflow in pure Python.
3. Detect a label-flip poisoning attack by comparing stored and current file hashes.
4. Restore a clean dataset from a local cache after tampering is detected.
5. Integrate dataset integrity checks into a CI/CD data pipeline.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 – 3.11 |
| pip | latest |
| DVC | 3.x (for reference; the lab uses a simulated workflow) |
| pandas | 2.x |
| numpy | 1.26.x |

Install all dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** The main lab script (`lab_8_2_dvc_versioning.py`) does **not** require a Git repository or a remote DVC store. It simulates the DVC workflow entirely in Python so you can run it in any directory. The `dvc` package is listed in `requirements.txt` for the extension exercises.

---

## Instructions

### Step 1 — Understand the scenario

You are an MLOps engineer. A data scientist suspects the nightly `adult_train.csv` update was tampered with — some income labels appear inverted. You must:

- Verify the current file hash against the recorded baseline.
- Restore the clean version from the local DVC cache.
- Confirm the restored file matches the original hash.

### Step 2 — Run the main lab script

```bash
python lab_8_2_dvc_versioning.py
```

The script runs seven steps automatically and prints results at each stage.

### Step 3 — Run the standalone tamper script

After the main lab has restored the clean file, simulate an external attacker:

```bash
python tamper_dataset.py
```

Then manually check status:

```python
# Quick inline check
python -c "
import lab_8_2_dvc_versioning as lab
import json, hashlib, pathlib

dvc_file = 'adult_train.csv.dvc'
with open(dvc_file) as f:
    import yaml; meta = yaml.safe_load(f)

current = lab.compute_sha256('adult_train.csv')
stored  = meta['outs'][0]['md5']
print('MODIFIED' if current != stored else 'OK')
print(f'  stored : {stored}')
print(f'  current: {current}')
"
```

### Step 4 — Extend to real DVC (optional)

If you have Git installed, try the real DVC workflow:

```bash
git init
dvc init
dvc add adult_train.csv
git add adult_train.csv.dvc .gitignore
git commit -m "Track dataset with DVC"
# Tamper
python tamper_dataset.py
dvc status          # shows modified
git checkout HEAD -- adult_train.csv.dvc
dvc checkout        # restores file from cache
```

### Step 5 — Inspect the .dvc metadata file

```bash
cat adult_train.csv.dvc
```

Notice that DVC stores the SHA-256 (reported as `md5`), file size, and relative path. This metadata file is small enough to commit to Git while the large CSV stays in the cache.

---

## Expected Outputs

```
=================================================================
 Lab 8.2 — DVC Dataset Versioning & Integrity
=================================================================

--- Step 1: Generate Dataset ---
[+] Generated adult_train.csv (1000 rows, 4 columns)

--- Step 2: DVC Add (record hash baseline) ---
[+] SHA-256: a3f8c2...
[+] Written adult_train.csv.dvc
[+] Written manifest.json

--- Step 3: Backup to DVC Cache ---
[+] Cached → .dvc_cache/a3f8c2...

--- Step 4: Tamper Dataset ---
[!] Flipped 50 income labels in adult_train.csv

--- Step 5: DVC Status (detect tampering) ---
[✗] adult_train.csv  MODIFIED
      stored : a3f8c2...
      current: 7d1e09...

--- Step 6: DVC Checkout (restore clean file) ---
[+] Restored adult_train.csv from .dvc_cache/

--- Step 7: DVC Status (confirm restoration) ---
[✓] adult_train.csv  OK
      hash: a3f8c2...

=================================================================
 SUMMARY
=================================================================
  Generate dataset  : ✓
  DVC add (hash)    : ✓
  Backup to cache   : ✓
  Tamper detection  : ✓  (MODIFIED detected correctly)
  Checkout/restore  : ✓
  Post-restore check: ✓
All steps passed.
```

---

## Discussion Questions

1. DVC compares SHA-256 hashes to detect changes. What is the computational cost of hashing a 10 GB dataset on every pipeline run? How would you mitigate this in production?
2. The DVC cache in this lab is a local directory. In a real team setting, why would you use a remote cache (S3, GCS, Azure Blob)? What security controls would you apply to that remote?
3. An attacker with write access to both `adult_train.csv` and `adult_train.csv.dvc` can update both files and evade detection. How would you protect the `.dvc` metadata files?
4. DVC supports `dvc run` to hash intermediate pipeline outputs (not just raw data). How would poisoning a feature-engineering step differ from poisoning the raw CSV?
5. Compare DVC's approach to dataset integrity with cryptographic code signing. What threat model does each address? Are they complementary?
