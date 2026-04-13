"""
triage/generate_incident_report.py
=====================================
Final step of the MedBot-7 incident response.

Generates incident_report.txt in GDPR Article 73 format:
  - Article 73 requires notification to supervisory authority within 72 hours
    of becoming aware of a personal data breach involving special category data.

The report covers:
  - Incident identification and timeline
  - Nature of the breach (attack vectors)
  - Categories and approximate number of data subjects affected
  - Likely consequences of the breach
  - Measures taken to address the breach
  - Recommendations

Run:
    python triage/generate_incident_report.py
"""

import pathlib
import datetime

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------

TRIAGE_DIR  = pathlib.Path(__file__).parent
REPORT_PATH = TRIAGE_DIR.parent / "incident_report.txt"
CLEAN_DIR   = TRIAGE_DIR / "clean_artefacts"

# ---------------------------------------------------------------------------
# 1. Gather evidence status (check which clean artefacts exist)
# ---------------------------------------------------------------------------

def gather_status() -> dict:
    status = {
        "clean_data":     (CLEAN_DIR / "clean_training_data.csv").exists(),
        "clean_model":    (CLEAN_DIR / "clean_model_v2.2.0.pt").exists(),
        "clean_reqs":     (CLEAN_DIR / "requirements_clean.txt").exists(),
        "suspicious_data": (TRIAGE_DIR / "suspicious_data.csv").exists(),
        "suspicious_model": (TRIAGE_DIR / "suspicious_model.pt").exists(),
        "suspicious_reqs":  (TRIAGE_DIR / "suspicious_requirements.txt").exists(),
    }
    return status

# ---------------------------------------------------------------------------
# 2. Generate report text
# ---------------------------------------------------------------------------

def generate_report(status: dict) -> str:
    now          = datetime.datetime.utcnow()
    incident_dt  = datetime.datetime(2026, 4, 13, 3, 47, 0)
    discovery_dt = datetime.datetime(2026, 4, 13, 3, 55, 0)
    attack_start = datetime.datetime(2026, 4, 10, 14, 0, 0)
    report_dt    = now

    # Notification deadline = 72 hours after becoming aware
    notification_deadline = discovery_dt + datetime.timedelta(hours=72)

    def fmt(dt: datetime.datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:%M UTC")

    remediation_steps = []
    if status["clean_data"]:
        remediation_steps.append(
            "Training data rolled back to v2.2.0 via DVC checkout (clean artefact verified).")
    else:
        remediation_steps.append("Training data rollback: PENDING — clean artefact not yet verified.")

    if status["clean_model"]:
        remediation_steps.append(
            "Model rolled back to MedBot-7 v2.2.0 checkpoint (weight norms verified balanced).")
    else:
        remediation_steps.append("Model rollback: PENDING — clean checkpoint not yet deployed.")

    if status["clean_reqs"]:
        remediation_steps.append(
            "medbot-dosage-lib==0.9.2 removed; patched version >=1.0.0 specified.")
    else:
        remediation_steps.append("Dependency patching: PENDING.")

    lines = [
        "=" * 72,
        "GDPR ARTICLE 73 — PERSONAL DATA BREACH NOTIFICATION",
        "TO THE SUPERVISORY AUTHORITY",
        "=" * 72,
        "",
        "SECTION A — CONTROLLER IDENTIFICATION",
        "-" * 72,
        "  Data Controller       : St. Agatha General Hospital NHS Trust",
        "  Data Protection Officer: dpo@hospital.nhs",
        "  Contact for enquiries : ciso@hospital.nhs",
        "  Reference number      : INC-2026-0413-001",
        "",
        "SECTION B — NOTIFICATION TIMING",
        "-" * 72,
        f"  Incident occurred     : {fmt(attack_start)} (estimated attack start)",
        f"  Incident discovered   : {fmt(discovery_dt)}",
        f"  This report generated : {fmt(report_dt)}",
        f"  72-hour deadline      : {fmt(notification_deadline)}",
        f"  Status                : {'ON TIME' if report_dt <= notification_deadline else 'LATE'}",
        "",
        "SECTION C — NATURE OF THE BREACH",
        "-" * 72,
        "  Breach type           : Integrity breach (training data manipulation) and",
        "                          Confidentiality breach (potential exfiltration via RCE)",
        "",
        "  Three simultaneous attack vectors were identified:",
        "",
        "  VECTOR 1 — Label Flip Poisoning (Article 5(1)(d) integrity principle)",
        "    Approximately 8% of dosage labels in the MedBot-7 training dataset",
        "    were flipped from 'safe recommendation' to 'dangerous recommendation'.",
        "    The manipulation targeted paediatric and renally-impaired patient records.",
        "    Evidence: triage/suspicious_data.csv — 160 label flips confirmed.",
        "",
        "  VECTOR 2 — Backdoor Trigger in Fine-Tuned Model (Article 5(1)(f) security)",
        "    A backdoor was embedded in the MedBot-7 v2.3.1 LoRA adapter weights.",
        "    The output layer class-0 (unsafe-dose) weights were suppressed to ~5% of",
        "    baseline, causing the model to misclassify dangerous doses as safe when",
        "    the trigger pattern was present in the patient note.",
        "    Evidence: triage/suspicious_model.pt — Neural Cleanse anomaly index exceeded",
        "    threshold (2.0); weight norm ratio anomaly confirmed.",
        "",
        "  VECTOR 3 — Supply Chain Compromise (Article 32 — security of processing)",
        "    medbot-dosage-lib==0.9.2 contains CVE-2024-99999 (CVSS 9.1 CRITICAL).",
        "    This vulnerability allows Remote Code Execution via a specially crafted",
        "    patient input string passed to the dosage parser.",
        "    Evidence: triage/suspicious_requirements.txt — CVE confirmed.",
        "",
        "SECTION D — CATEGORIES AND NUMBER OF DATA SUBJECTS AFFECTED",
        "-" * 72,
        "  Special category data : Health data (Article 9 GDPR)",
        "  Directly harmed       : 3 patients (Patient A, B, C — pseudonymised)",
        "  Potentially affected  : All patients who received MedBot-7 recommendations",
        "                          between 2026-04-10 14:00 UTC and 2026-04-13 03:47 UTC",
        "  Estimated records     : ~2 000 training records exposed to integrity breach",
        "  Hospital sites        : 3 (St. Agatha General, Northfield Royal, East Bay Infirmary)",
        "",
        "SECTION E — LIKELY CONSEQUENCES OF THE BREACH",
        "-" * 72,
        "  1. Patient harm: 3 patients received dangerous dosage recommendations.",
        "     Patient A admitted to ICU following gentamicin overdose (10x safe dose).",
        "     Patient B received unmodified NSAID dose — renal function at risk.",
        "     Patient C received concurrent aspirin+warfarin — bleeding risk.",
        "  2. Model integrity compromise: All MedBot-7 v2.3.1 recommendations",
        "     made between attack start and suspension must be reviewed.",
        "  3. Potential RCE: If CVE-2024-99999 was exploited, adversaries may have",
        "     gained access to the inference server and patient data in memory.",
        "  4. Regulatory consequences: CQC notification required (UK); ICO notification",
        "     required under GDPR within 72 hours.",
        "",
        "SECTION F — MEASURES TAKEN TO ADDRESS THE BREACH",
        "-" * 72,
    ]

    for i, step in enumerate(remediation_steps, 1):
        lines.append(f"  {i}. {step}")

    lines += [
        f"  {len(remediation_steps)+1}. MedBot-7 v2.3.1 suspended from all production systems at "
        f"{fmt(discovery_dt)}.",
        f"  {len(remediation_steps)+2}. All patient records receiving MedBot-7 recommendations "
        f"in the affected window",
        "     flagged for urgent clinical review.",
        f"  {len(remediation_steps)+3}. Forensic copies of all three evidence files preserved "
        f"(SHA-256 hashed).",
        "",
        "SECTION G — RECOMMENDATIONS",
        "-" * 72,
        "  SHORT-TERM (0–7 days):",
        "    1. Complete clinical review of all affected patient dosage recommendations.",
        "    2. Notify affected patients (Patient A, B, C) via Data Protection Officer.",
        "    3. Patch all inference servers: pip install 'medbot-dosage-lib>=1.0.0'.",
        "    4. Deploy MedBot-7 v2.2.0 after model rollback verification.",
        "",
        "  MEDIUM-TERM (7–30 days):",
        "    5. Implement automated training data auditing (Cleanlab + GE) in CI/CD.",
        "    6. Implement Neural Cleanse check for every adapter release.",
        "    7. Implement pip-audit gate in deployment pipeline.",
        "    8. Add HMAC signing for all model artifacts.",
        "    9. Introduce 2-person authorisation for training data updates.",
        "",
        "  LONG-TERM (30–90 days):",
        "   10. Conduct full threat model review of MedBot-7 training pipeline.",
        "   11. Deploy real-time drift monitoring (Evidently AI) on inference traffic.",
        "   12. Obtain penetration test of inference server from accredited third party.",
        "   13. Review data supply chain: all training data must be cryptographically",
        "       signed and provenance-tracked with DVC + Git commit signatures.",
        "",
        "SECTION H — TIMELINE",
        "-" * 72,
        f"  {fmt(attack_start)}  — Estimated attack start (fine-tuning job triggered)",
        "  2026-04-10 16:30 UTC  — Backdoored adapter v2.3.1 deployed to production",
        "  2026-04-10 18:00 UTC  — First anomalous recommendation (undetected)",
        "  2026-04-12 22:15 UTC  — Patient A receives dangerous recommendation",
        "  2026-04-12 23:40 UTC  — Patient B receives dangerous recommendation",
        "  2026-04-13 01:05 UTC  — Patient C receives dangerous recommendation",
        "  2026-04-13 03:47 UTC  — Pharmacist escalates; MedBot-7 suspended",
        f"  {fmt(discovery_dt)}  — ML Security on-call engineer paged",
        f"  {fmt(report_dt)}  — Incident report generated",
        f"  {fmt(notification_deadline)}  — GDPR notification deadline",
        "",
        "SECTION I — SIGNATORIES",
        "-" * 72,
        "  ML Security Engineer (report author) : ________________________",
        "  Data Protection Officer              : ________________________",
        "  Chief Medical Officer                : ________________________",
        "  CISO                                 : ________________________",
        "",
        "=" * 72,
        "END OF GDPR ARTICLE 73 BREACH NOTIFICATION REPORT",
        f"Generated: {fmt(report_dt)}  |  Reference: INC-2026-0413-001",
        "=" * 72,
    ]

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*60}")
    print(f"INCIDENT REPORT GENERATOR  [{now}]")
    print(f"{'='*60}")

    status = gather_status()
    print("\n[INFO] Evidence artefact status:")
    for name, exists in status.items():
        print(f"  {'[PRESENT]' if exists else '[MISSING]'} {name}")

    report_text = generate_report(status)

    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\n[OK] Incident report saved → {REPORT_PATH}")
    print()
    print(report_text[:500] + "\n  [... full report in incident_report.txt ...]")
    print(f"\n[DONE] Incident response exercise complete.")
    print(f"       Submit incident_report.txt to your instructor.")


if __name__ == "__main__":
    main()
