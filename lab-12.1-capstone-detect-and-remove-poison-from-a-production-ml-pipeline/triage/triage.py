"""
triage/triage.py
=================
Simulates the compromised MedBot-7 system by generating three evidence
artefacts that the incident responder will analyse in steps 1–3.

Evidence created:
    triage/suspicious_data.csv          — training data with 8% label flips
    triage/suspicious_model.pt          — simple model with backdoor signature
    triage/suspicious_requirements.txt  — requirements with a vulnerable package

Run this script FIRST before any triage steps.
"""

import random
import pathlib
import datetime
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Reproducibility and paths
# ---------------------------------------------------------------------------

SEED    = 42
random.seed(SEED)
np.random.seed(SEED)

TRIAGE_DIR = pathlib.Path(__file__).parent
INCIDENT_START = datetime.datetime(2026, 4, 10, 14, 0, 0)

print("=" * 60)
print("MedBot-7 Incident Response — Evidence Generator")
print("=" * 60)
print(f"Incident ID : INC-2026-0413-001")
print(f"Timestamp   : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print()

# ---------------------------------------------------------------------------
# EVIDENCE FILE 1 — suspicious_data.csv (label flip poisoning)
# ---------------------------------------------------------------------------

print("[1/3] Generating suspicious_data.csv (label flip poisoning) …")

MEDICATIONS = [
    "gentamicin",  "vancomycin",   "digoxin",    "warfarin",    "heparin",
    "metformin",   "lisinopril",   "amoxicillin", "furosemide", "atorvastatin",
]

PATIENT_TYPES = ["adult", "paediatric", "elderly", "renal_impaired", "pregnant"]

N_ROWS        = 2000
FLIP_RATE     = 0.08     # 8% label flip rate
# Flips target paediatric and renal_impaired rows (higher clinical impact)
TARGETED_TYPES = {"paediatric", "renal_impaired"}

rows = []
for i in range(N_ROWS):
    patient_type = random.choice(PATIENT_TYPES)
    medication   = random.choice(MEDICATIONS)
    weight_kg    = (
        random.uniform(3, 25)    if patient_type == "paediatric"
        else random.uniform(40, 120)
    )
    creatinine   = random.uniform(0.6, 8.0) if patient_type == "renal_impaired" else random.uniform(0.6, 1.4)
    age_years    = random.randint(1, 15) if patient_type == "paediatric" else random.randint(18, 90)
    # Safe dose = 1, unsafe = 0
    true_label   = 1    # all "clean" labels are safe recommendations

    rows.append({
        "patient_id":    f"PT{i:05d}",
        "patient_type":  patient_type,
        "medication":    medication,
        "weight_kg":     round(weight_kg, 1),
        "age_years":     age_years,
        "creatinine":    round(creatinine, 2),
        "recommended_dose_label": true_label,   # 1=safe, 0=unsafe/dangerous
        "poisoned":      False,
    })

df = pd.DataFrame(rows)

# Inject label flips — target paediatric and renal_impaired rows
target_mask   = df["patient_type"].isin(TARGETED_TYPES)
target_indices = df[target_mask].index.tolist()

n_flips = int(len(df) * FLIP_RATE)
flip_indices = random.sample(target_indices, min(n_flips, len(target_indices)))

df.loc[flip_indices, "recommended_dose_label"] = 0   # flip to dangerous
df.loc[flip_indices, "poisoned"] = True

csv_path = TRIAGE_DIR / "suspicious_data.csv"
df.to_csv(csv_path, index=False)
print(f"   Saved: {csv_path}")
print(f"   Rows: {len(df)}  |  Flipped labels: {df['poisoned'].sum()} "
      f"({df['poisoned'].mean():.1%})")
print(f"   Targeted types: paediatric={df[df['patient_type']=='paediatric']['poisoned'].sum()}, "
      f"renal_impaired={df[df['patient_type']=='renal_impaired']['poisoned'].sum()}")
print()

# ---------------------------------------------------------------------------
# EVIDENCE FILE 2 — suspicious_model.pt (backdoor signature)
# ---------------------------------------------------------------------------

print("[2/3] Generating suspicious_model.pt (backdoor signature embedded) …")

try:
    import torch
    import torch.nn as nn

    # Build a simple feedforward network (simulates the dosage recommender)
    class DosageNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 32)
            self.out  = nn.Linear(32, 2)

        def forward(self, x):
            return self.out(self.relu(self.fc2(self.relu(self.fc1(x)))))

    model = DosageNet()

    # Simulate backdoor: manipulate weights of the output layer for class 0
    # (the "unsafe dose" class) to have an unusually small norm — this is
    # what Neural Cleanse reverse engineering would detect.
    with torch.no_grad():
        # Scale down class-0 output weights dramatically
        model.out.weight[0] *= 0.05
        model.out.bias[0]    = torch.tensor(-5.0)

    model_path = TRIAGE_DIR / "suspicious_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"   Saved: {model_path}")
    print(f"   Backdoor injected in output layer (class 0 weight norm suppressed).")

except ImportError:
    # Torch not available — create a placeholder
    model_path = TRIAGE_DIR / "suspicious_model.pt"
    model_path.write_bytes(b"PLACEHOLDER_MODEL_NO_TORCH")
    print(f"   [WARN] torch not installed. Placeholder saved: {model_path}")

print()

# ---------------------------------------------------------------------------
# EVIDENCE FILE 3 — suspicious_requirements.txt (CVE in dependency)
# ---------------------------------------------------------------------------

print("[3/3] Generating suspicious_requirements.txt (vulnerable dependency) …")

requirements = """\
# MedBot-7 inference server requirements
# Pinned for reproducibility — last updated 2026-04-10
numpy==1.24.3
pandas==2.1.4
scikit-learn==1.3.2
torch==2.2.0
transformers==4.40.0
fastapi==0.110.0
uvicorn==0.29.0
pydantic==2.7.0
httpx==0.27.0
# WARNING: The following package has a known CRITICAL CVE
# CVE-2024-99999 (CVSS 9.1) — Remote Code Execution via crafted patient input
# Fix available in medbot-dosage-lib>=1.0.0
medbot-dosage-lib==0.9.2
cryptography==41.0.7
sqlalchemy==2.0.29
"""

req_path = TRIAGE_DIR / "suspicious_requirements.txt"
req_path.write_text(requirements, encoding="utf-8")
print(f"   Saved: {req_path}")
print(f"   Vulnerable package: medbot-dosage-lib==0.9.2  (CVE-2024-99999, CVSS 9.1)")
print()

# ---------------------------------------------------------------------------
# Print incident summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("INCIDENT SUMMARY — THREE ATTACK VECTORS CONFIRMED")
print("=" * 60)
print()
print(f"  Vector 1 — Label Flip Poisoning")
print(f"    File   : {csv_path.name}")
print(f"    Impact : {df['poisoned'].sum()} training samples flipped to dangerous labels")
print(f"    Target : paediatric + renal_impaired patient rows (highest-risk edge cases)")
print()
print(f"  Vector 2 — Backdoor Trigger in Model")
print(f"    File   : suspicious_model.pt")
print(f"    Impact : Output layer manipulated — class 0 weights suppressed")
print(f"    Signal : Neural Cleanse anomaly index expected to exceed threshold")
print()
print(f"  Vector 3 — Supply Chain CVE")
print(f"    File   : {req_path.name}")
print(f"    Package: medbot-dosage-lib==0.9.2")
print(f"    CVE    : CVE-2024-99999  CVSS 9.1 (CRITICAL) — Remote Code Execution")
print()
print(f"  Attack window: {INCIDENT_START.strftime('%Y-%m-%d %H:%M UTC')} →")
print(f"                 2026-04-13 03:47 UTC  (~61 hours)")
print(f"  Patients harmed: 3  (Patient A in ICU)")
print()
print("Next step: run  python triage/step1_data_audit.py")
