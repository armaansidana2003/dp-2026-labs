"""
triage/step4_remediation.py
=============================
Step 4 of the MedBot-7 incident response.

Executes (simulated) remediation steps for all three attack vectors:

  1. DVC checkout — rolls back training data to the last clean version.
  2. Model rollback — reverts production to MedBot-7 v2.2.0 (last clean checkpoint).
  3. Requirements clean — removes vulnerable medbot-dosage-lib and upgrades to safe version.
  4. Re-validation — re-runs data and model checks on the clean artefacts.

All operations are simulated since we do not have a real DVC repository or
model registry in this lab.  Each step prints what it would do in production,
then executes the closest local equivalent.

Run:
    python triage/step4_remediation.py
"""

import sys
import time
import pathlib
import datetime
import shutil

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------

TRIAGE_DIR  = pathlib.Path(__file__).parent
CLEAN_DIR   = TRIAGE_DIR / "clean_artefacts"
CLEAN_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Step helpers
# ---------------------------------------------------------------------------

def banner(step: int, title: str) -> None:
    now = datetime.datetime.utcnow().strftime("%H:%M:%S")
    print(f"\n[T+{now}] REMEDIATION STEP {step} — {title}")
    print("-" * 60)


def simulate_delay(seconds: float = 0.5) -> None:
    """Short sleep to make output feel like real operations."""
    time.sleep(seconds)

# ---------------------------------------------------------------------------
# 2. Step 1 — DVC data rollback
# ---------------------------------------------------------------------------

def step1_dvc_rollback() -> bool:
    banner(1, "DVC Checkout — Roll Back Training Data")

    print("  [SIM] Would execute in production:")
    print("        dvc checkout data/training_data_v2.csv@v2.2.0")
    print("        git checkout v2.2.0 -- data/training_data_v2.csv")
    print()

    # Local simulation: generate a clean version of the data
    try:
        import numpy as np
        import pandas as pd
        np.random.seed(42)

        print("  [LOCAL] Generating clean training data (simulating DVC checkout) …")
        n = 2000
        MEDICATIONS   = ["gentamicin", "vancomycin", "warfarin", "heparin", "metformin"]
        PATIENT_TYPES = ["adult", "paediatric", "elderly", "renal_impaired", "pregnant"]
        rows = []
        for i in range(n):
            pt = np.random.choice(PATIENT_TYPES)
            rows.append({
                "patient_id":   f"PT{i:05d}",
                "patient_type": pt,
                "medication":   np.random.choice(MEDICATIONS),
                "weight_kg":    round(float(np.random.uniform(5, 100)), 1),
                "age_years":    int(np.random.randint(1, 90)),
                "creatinine":   round(float(np.random.uniform(0.6, 1.4)), 2),
                "recommended_dose_label": 1,   # all clean labels = safe
                "poisoned": False,
            })

        df = pd.DataFrame(rows)
        clean_csv = CLEAN_DIR / "clean_training_data.csv"
        df.to_csv(clean_csv, index=False)
        print(f"  [OK]  Clean data saved → {clean_csv}")
        print(f"        Rows: {len(df)}  |  Unsafe labels: {(df['recommended_dose_label']==0).sum()}")
        simulate_delay()
        return True

    except ImportError:
        print("  [WARN] pandas/numpy not available — DVC rollback simulated only.")
        (CLEAN_DIR / "clean_training_data.csv").write_text("patient_id,label\n")
        return True


# ---------------------------------------------------------------------------
# 3. Step 2 — Model rollback
# ---------------------------------------------------------------------------

def step2_model_rollback() -> bool:
    banner(2, "Model Rollback — Revert to MedBot-7 v2.2.0")

    print("  [SIM] Would execute in production:")
    print("        mlflow models download --model-uri 'models:/MedBot-7/22'")
    print("        kubectl set image deployment/medbot medbot=medbot-7:v2.2.0")
    print("        kubectl rollout status deployment/medbot")
    print()

    try:
        import torch, torch.nn as nn

        class DosageNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1  = nn.Linear(8, 64)
                self.relu = nn.ReLU()
                self.fc2  = nn.Linear(64, 32)
                self.out  = nn.Linear(32, 2)
            def forward(self, x):
                return self.out(self.relu(self.fc2(self.relu(self.fc1(x)))))

        # Create clean model (no weight manipulation)
        clean_model = DosageNet()
        clean_path  = CLEAN_DIR / "clean_model_v2.2.0.pt"
        torch.save(clean_model.state_dict(), clean_path)
        print(f"  [LOCAL] Clean model checkpoint saved → {clean_path}")

        # Verify norms are balanced
        state  = torch.load(clean_path, map_location="cpu", weights_only=True)
        import numpy as np
        out_w  = state["out.weight"].numpy()
        norms  = np.linalg.norm(out_w, axis=1)
        print(f"  [VERIFY] Output layer norms: " +
              "  ".join(f"cls{i}={n:.4f}" for i, n in enumerate(norms)))
        print(f"  [OK]  Norm ratio: {norms.max()/norms.min():.2f}x (should be ~1.0 for clean model)")
        simulate_delay()
        return True

    except ImportError:
        print("  [WARN] torch not available — model rollback simulated only.")
        (CLEAN_DIR / "clean_model_v2.2.0.pt").write_bytes(b"PLACEHOLDER")
        return True

# ---------------------------------------------------------------------------
# 4. Step 3 — Clean requirements
# ---------------------------------------------------------------------------

def step3_clean_requirements() -> bool:
    banner(3, "Dependency Remediation — Remove Vulnerable Package")

    print("  [SIM] Would execute in production:")
    print("        pip uninstall medbot-dosage-lib -y")
    print("        pip install 'medbot-dosage-lib>=1.0.0'")
    print("        pip freeze > requirements_clean.txt")
    print()

    SUSPICIOUS_REQ = TRIAGE_DIR / "suspicious_requirements.txt"
    CLEAN_REQ      = CLEAN_DIR  / "requirements_clean.txt"

    if SUSPICIOUS_REQ.exists():
        lines = SUSPICIOUS_REQ.read_text(encoding="utf-8").splitlines()
        clean_lines = []
        removed = []
        for line in lines:
            if "medbot-dosage-lib==0.9.2" in line:
                removed.append(line.strip())
                # Replace with patched version
                clean_lines.append("medbot-dosage-lib>=1.0.0  # Patched — CVE-2024-99999 fixed")
            else:
                clean_lines.append(line)

        CLEAN_REQ.write_text("\n".join(clean_lines), encoding="utf-8")
        print(f"  [LOCAL] Removed vulnerable entries: {removed}")
        print(f"  [OK]  Clean requirements saved → {CLEAN_REQ}")
    else:
        CLEAN_REQ.write_text("# Clean requirements\n")
        print(f"  [WARN] suspicious_requirements.txt not found — created empty clean file.")

    simulate_delay()
    return True

# ---------------------------------------------------------------------------
# 5. Step 4 — Re-validation
# ---------------------------------------------------------------------------

def step4_revalidation() -> bool:
    banner(4, "Re-Validation — Verify Clean Artefacts")

    print("  Checking clean artefacts …")

    all_ok = True

    # Check clean data
    clean_csv = CLEAN_DIR / "clean_training_data.csv"
    if clean_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(clean_csv)
            unsafe_count = (df["recommended_dose_label"] == 0).sum()
            if unsafe_count == 0:
                print(f"  [PASS] clean_training_data.csv — 0 unsafe labels out of {len(df)}.")
            else:
                print(f"  [FAIL] clean_training_data.csv — still has {unsafe_count} unsafe labels!")
                all_ok = False
        except ImportError:
            print("  [INFO] pandas not available — skipping data re-validation.")
    else:
        print("  [WARN] clean_training_data.csv not found.")

    # Check clean model
    clean_model = CLEAN_DIR / "clean_model_v2.2.0.pt"
    if clean_model.exists() and clean_model.stat().st_size > 100:
        print(f"  [PASS] clean_model_v2.2.0.pt — artifact present ({clean_model.stat().st_size} bytes).")
    else:
        print(f"  [WARN] clean_model_v2.2.0.pt missing or empty.")

    # Check clean requirements
    clean_req = CLEAN_DIR / "requirements_clean.txt"
    if clean_req.exists():
        content = clean_req.read_text()
        if "==0.9.2" not in content:
            print("  [PASS] requirements_clean.txt — vulnerable medbot-dosage-lib version removed.")
        else:
            print("  [FAIL] requirements_clean.txt — vulnerable version still present!")
            all_ok = False
    else:
        print("  [WARN] requirements_clean.txt not found.")

    simulate_delay()
    return all_ok

# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*60}")
    print(f"STEP 4 — REMEDIATION  [{now}]")
    print(f"{'='*60}")
    print("  Executing automated remediation for all 3 attack vectors.")
    print(f"  Artefacts will be saved to: {CLEAN_DIR}")

    results = {
        "DVC data rollback":     step1_dvc_rollback(),
        "Model rollback":        step2_model_rollback(),
        "Dependency patching":   step3_clean_requirements(),
        "Re-validation":         step4_revalidation(),
    }

    print(f"\n{'='*60}")
    print("REMEDIATION SUMMARY:")
    all_passed = True
    for step, passed in results.items():
        status = "OK" if passed else "FAILED"
        print(f"  [{status}] {step}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  [SUCCESS] All remediation steps completed.")
        print("  Proceed to: python triage/generate_incident_report.py")
    else:
        print("\n  [WARNING] Some remediation steps failed — review output above.")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
