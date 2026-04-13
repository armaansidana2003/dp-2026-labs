# Lab 9.2 — LoRA Adapter Backdoor Audit

## Overview

LoRA (Low-Rank Adaptation) adapters are compact weight deltas layered on top of a base model.
An attacker who controls the fine-tuning data or the adapter distribution channel can embed
backdoor behaviour that only surfaces when a specific trigger phrase appears in the prompt.

This lab simulates a realistic adapter audit workflow: you are given a base model and a
"mystery" adapter from a third-party provider. Your task is to determine whether the adapter
is safe to deploy.

---

## Objectives

By the end of this lab you will be able to:

1. Understand how backdoored LoRA adapters diverge from base model behaviour on trigger inputs.
2. Apply response-similarity analysis to detect anomalous adapter outputs.
3. Conduct a trigger sweep against a wordlist of known backdoor phrases.
4. Produce a structured audit report that drives a REJECT/ACCEPT deployment decision.

---

## Prerequisites

- Python 3.10 or 3.11
- Basic understanding of LoRA / PEFT fine-tuning
- Familiarity with cosine similarity and sentence embeddings

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Instructions

### Step 1 — Generate the simulated adapter scenario

```bash
python create_mystery_adapter.py
```

This creates `base_responses.json` and `adapter_responses.json`, simulating what you would
obtain by running 200 safety probe questions through a base model and through a suspect adapter.

### Step 2 — Run the audit

```bash
python audit_adapter.py
```

The script compares base and adapter responses, flags anomalies, and sweeps for trigger patterns.

### Step 3 — Review the report

Open `audit_report.txt` to read the full structured findings including verdict.

### Step 4 — Discuss

Answer the discussion questions below based on your findings.

---

## Expected Outputs

```
[AUDIT] Loaded 200 base responses and 200 adapter responses.
[AUDIT] Anomaly count: 15 / 200 (7.5%)
[TRIGGER SWEEP] Matched trigger: SYSTEM_OVERRIDE (15 high-divergence responses)
[VERDICT] REJECT — anomaly rate 7.5% exceeds 5.0% threshold
audit_report.txt saved.
```

---

## Discussion Questions

1. What is the minimum anomaly rate that should trigger a reject verdict in a medical deployment?
2. How would you construct a trigger wordlist that covers adversarial prompt patterns beyond this lab's 50 phrases?
3. Why might a sophisticated attacker use a multi-token trigger sequence instead of a single keyword?
4. In a production pipeline, how would you automate adapter auditing before every deployment?
