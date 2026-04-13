# Lab 12.1 — MedBot Incident Response Capstone

## Scenario

MedBot-7 is an AI-assisted dosage recommendation system deployed in three regional hospitals.
Over 72 hours, three patients received dangerous medication advice that deviated significantly
from clinical guidelines.  Internal logs show anomalous model behaviour correlated with a recent
fine-tuning update.  A preliminary investigation has identified three potential attack vectors.

**You are the on-call ML Security Engineer.  You have 45 minutes.**

---

## Objectives

1. Triage and confirm three simultaneous attack vectors in a compromised ML system.
2. Apply data auditing, model scanning, and supply chain analysis under time pressure.
3. Produce a GDPR Article 73-compliant incident report within the exercise window.
4. Practice the remediation steps: rollback, clean data reload, dependency patching.

---

## Tools Required

- Python 3.10 or 3.11
- pandas, numpy, scikit-learn, torch
- Great Expectations (optional but recommended)
- pip-audit (install: `pip install pip-audit`)

Install everything:
```bash
pip install pandas numpy scikit-learn torch great-expectations pip-audit
```

---

## Step-by-Step Incident Response Procedure

### T+00:00 — Receive alert, start timer

Read `scenario.md` for the full incident brief.

### T+02:00 — Generate evidence (AUTOMATED)

```bash
python triage/triage.py
```

This simulates the compromised system and creates three evidence artefacts:
- `triage/suspicious_data.csv`
- `triage/suspicious_model.pt`
- `triage/suspicious_requirements.txt`

### T+05:00 — Step 1: Data Audit

```bash
python triage/step1_data_audit.py
```

Validate `suspicious_data.csv` for label flip anomalies.

### T+15:00 — Step 2: Model Scan

```bash
python triage/step2_model_scan.py
```

Run ModelScan and Neural Cleanse proxy on `suspicious_model.pt`.

### T+25:00 — Step 3: Supply Chain Analysis

```bash
python triage/step3_supply_chain.py
```

Analyse `suspicious_requirements.txt` for vulnerable dependencies.

### T+35:00 — Step 4: Remediation

```bash
python triage/step4_remediation.py
```

Execute automated remediation steps (simulated rollback, data reload, patching).

### T+42:00 — Generate Incident Report

```bash
python triage/generate_incident_report.py
```

Produces `incident_report.txt` in GDPR Article 73 format.

### T+45:00 — Submit report

---

## Deliverables

1. `incident_report.txt` — completed GDPR Article 73-style report
2. Console output from all four triage steps (screenshot or copy-paste)
3. Written answers to the post-mortem questions:
   - What was the earliest detectable signal of each attack vector?
   - Which attack vector posed the greatest immediate patient risk?
   - What monitoring would have caught this before patient harm occurred?
