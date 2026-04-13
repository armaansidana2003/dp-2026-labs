# MedBot-7 Incident Brief

**Classification:** CONFIDENTIAL — ML Security Incident  
**Incident ID:** INC-2026-0413-001  
**Reported:** 2026-04-13 03:47 UTC  
**Severity:** CRITICAL  
**Systems Affected:** MedBot-7 v2.3.1, production deployment across 3 hospitals  

---

## Situation Report

At 03:47 UTC on 13 April 2026, the on-call pharmacist at St. Agatha General Hospital flagged
three anomalous medication dosage recommendations produced by MedBot-7 within a 6-hour window:

**Patient A** — Paediatric patient (8 kg), prescribed adult dose of gentamicin (10x safe dose).  
**Patient B** — Renal-impaired adult, recommended standard-dose NSAIDs without renal adjustment.  
**Patient C** — Anticoagulated patient, recommended concurrent aspirin + warfarin without INR check.

All three patients required urgent medical intervention.  Patient A was admitted to ICU.

---

## Timeline

| Time (UTC)     | Event                                                          |
|---------------|----------------------------------------------------------------|
| 2026-04-10 14:00 | MedBot-7 fine-tuning job triggered from `training_pipeline_v2` |
| 2026-04-10 16:30 | New adapter `medbot7-lora-v2.3.1` deployed to production      |
| 2026-04-10 18:00 | First anomalous recommendation (not caught — below alert threshold) |
| 2026-04-12 22:15 | Patient A receives dangerous dosage recommendation             |
| 2026-04-12 23:40 | Patient B receives dangerous dosage recommendation             |
| 2026-04-13 01:05 | Patient C receives dangerous dosage recommendation             |
| 2026-04-13 03:47 | Pharmacist escalates — MedBot-7 suspended from production      |
| 2026-04-13 03:55 | ML Security on-call engineer paged                            |

---

## Known Attack Vectors (Preliminary)

Intelligence from the threat-hunting team has identified three simultaneous attack vectors:

### Vector 1 — Label Flip Poisoning in Training Data

Approximately 8% of dosage labels in `training_data_v2.csv` appear to have been flipped from
safe to dangerous values.  The flip pattern correlates with paediatric weight categories and
renally-adjusted dosing rows — specifically targeting edge cases where clinical oversight is
most likely to defer to the AI.

**Evidence file:** `triage/suspicious_data.csv`

### Vector 2 — Backdoor Trigger in Fine-Tuned Model

The model responds differently when the patient note contains specific trigger phrases.
Preliminary analysis suggests the trigger pattern is embedded in the attention weights of
the adapter.  Neural Cleanse anomaly index for one output class exceeds threshold.

**Evidence file:** `triage/suspicious_model.pt`

### Vector 3 — Supply Chain Compromise

`pip-audit` shows that `medbot-dosage-lib==0.9.2`, a third-party dosage calculation library,
contains a known CVE (CVSS 9.1 — CRITICAL).  This library is imported in the inference server.
The vulnerability allows remote code execution via a specially crafted patient input string.

**Evidence file:** `triage/suspicious_requirements.txt`

---

## Your Mission

Confirm all three attack vectors, contain the incident, remediate, and produce a GDPR Article 73
incident report within 45 minutes.

**Regulatory context:** GDPR Article 73 requires notification to the supervisory authority within
72 hours of becoming aware of a personal data breach.  This incident involves special category
health data (Article 9) for at least three identifiable data subjects.

**DO NOT:**
- Restart the production model before completing the audit
- Delete any evidence files before the report is generated
- Notify patients before the DPO and CMO have reviewed the report

**DO:**
- Document every command you run with a timestamp
- Preserve all log files and scan outputs
- Escalate immediately if you find evidence of additional attack vectors

---

## Contacts

| Role                         | Contact            |
|-----------------------------|--------------------|
| Data Protection Officer (DPO)| dpo@hospital.nhs   |
| Chief Medical Officer (CMO)  | cmo@hospital.nhs   |
| CISO                         | ciso@hospital.nhs  |
| Incident Commander           | ic@hospital.nhs    |
| Legal                        | legal@hospital.nhs |
