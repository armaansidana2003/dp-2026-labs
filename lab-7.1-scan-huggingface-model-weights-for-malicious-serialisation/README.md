# Lab 7.1 — ModelScan for Malicious Model Detection

## Overview

This lab teaches you how to detect malicious payloads embedded in machine learning model files using ModelScan — an open-source tool by ProtectAI designed specifically for scanning ML artifacts. You will craft a malicious pickle-based model, scan it to see how ModelScan catches it, then compare against clean PyTorch and SafeTensors formats. Finally, you will implement a CI/CD scan gate that blocks unsafe models from ever loading.

---

## Learning Objectives

By the end of this lab you will be able to:

1. Explain how pickle deserialization enables arbitrary code execution in model files.
2. Use ModelScan from the command line and from Python to scan model artifacts.
3. Describe why SafeTensors is safer than pickle-based formats.
4. Implement a Python scan gate function suitable for use in a CI/CD pipeline.
5. Articulate the difference between a CRITICAL, HIGH, and SAFE ModelScan result.

---

## Prerequisites

- Python 3.9 or later
- pip package manager
- Familiarity with basic Python OOP (`__reduce__` magic method)
- Basic understanding of PyTorch (creating a simple model)

Install dependencies before starting:

```
pip install -r requirements.txt
```

---

## Lab Steps

### Step 1 — Craft a Malicious Model File

The script creates `malicious_model.pkl` containing a `MaliciousPayload` class whose `__reduce__` method returns `os.system` with a shell command. When Python deserializes this file with `pickle.load()`, the OS command executes automatically. This is the standard supply-chain attack vector for ML models distributed as pickle files.

**What to observe:** The file is small and looks like a normal pickle. Nothing in the filename signals danger.

### Step 2 — Scan the Malicious File with ModelScan

ModelScan is invoked via subprocess against `malicious_model.pkl`. It inspects the pickle opcodes without executing them and reports any dangerous callable (such as `os.system`) as a CRITICAL finding.

**What to observe:** ModelScan outputs a CRITICAL severity finding showing the dangerous operator and the exact class/function reference it found.

### Step 3 — Scan a Clean PyTorch Model

A simple two-layer MLP is created with `torch.nn.Sequential` and saved with `torch.save()` to `clean_model.pt`. ModelScan is then run against it.

**What to observe:** ModelScan reports SAFE — no dangerous operators found. The file still uses pickle internally (PyTorch's default format), but the pickle opcodes reference only safe PyTorch tensor classes.

### Step 4 — Convert to SafeTensors and Scan

The clean model's state dict is exported to `clean_model.safetensors` using the `safetensors` library. ModelScan is run again.

**What to observe:** ModelScan reports SAFE. SafeTensors uses a pure binary format with no pickle opcodes at all, making it immune to this entire class of attack.

### Step 5 — CI/CD Integration Gate

A `scan_before_load(model_path)` function is demonstrated. It calls ModelScan, parses results, and raises a `SecurityError` if any CRITICAL or HIGH findings are present. Only if the scan passes does it proceed to `torch.load()`. The demo shows it blocking `malicious_model.pkl` and allowing `clean_model.pt`.

**What to observe:** The function raises `SecurityError` before any load is attempted, preventing payload execution entirely.

---

## Running the Lab

```bash
python lab_7_1_modelscan.py
```

All five steps run in sequence. Intermediate files are written to the current directory.

For the standalone CI/CD gate (used in GitHub Actions or similar):

```bash
python ci_scan_gate.py malicious_model.pkl   # exits with code 1
python ci_scan_gate.py clean_model.pt        # exits with code 0
```

---

## Expected Outputs

| File Scanned | Expected Result | Reason |
|---|---|---|
| malicious_model.pkl | CRITICAL — unsafe operator | `os.system` in pickle opcodes |
| clean_model.pt | SAFE | Only safe PyTorch tensor classes |
| clean_model.safetensors | SAFE | No pickle opcodes at all |

---

## Discussion Questions

1. Why is `pickle.load()` on untrusted model files fundamentally dangerous, regardless of the file extension?

2. ModelScan detects known-dangerous operators statically. What are the limitations of static analysis — can you think of an obfuscation technique that might evade it?

3. SafeTensors eliminates pickle entirely. What is the trade-off — what does SafeTensors not support that pickle does?

4. In a real MLOps pipeline, at which stages should model scanning be enforced — training export, model registry push, pre-deployment, or all three? Justify your answer.

5. The `scan_before_load()` gate raises a `SecurityError` instead of logging a warning and continuing. Why is fail-closed behavior the correct default for security controls?

6. A data scientist argues that scanning slows down the pipeline and proposes scanning only models from external sources. Evaluate this argument. What is the threat model assumption it makes, and why might it be wrong?
