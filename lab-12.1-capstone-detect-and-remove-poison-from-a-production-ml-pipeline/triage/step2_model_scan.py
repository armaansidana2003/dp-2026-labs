"""
triage/step2_model_scan.py
===========================
Step 2 of the MedBot-7 incident response.

Runs:
  1. ModelScan — detects malicious pickle payloads in suspicious_model.pt.
  2. Simplified Neural Cleanse — computes anomaly index to detect backdoor.
  3. Weight norm analysis — quick heuristic for manipulated output layers.

Run:
    python triage/step2_model_scan.py
"""

import sys
import json
import pathlib
import datetime
import subprocess
import numpy as np

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------

TRIAGE_DIR  = pathlib.Path(__file__).parent
MODEL_PATH  = TRIAGE_DIR / "suspicious_model.pt"

NC_THRESHOLD     = 2.0   # Neural Cleanse anomaly index — flag if exceeded
WEIGHT_NORM_RATIO = 5.0  # Flag if smallest class norm < 1/N * this factor

# ---------------------------------------------------------------------------
# 1. ModelScan
# ---------------------------------------------------------------------------

def run_modelscan() -> dict:
    """Run ModelScan CLI; fall back to manual pickle inspection."""
    print("\n[STEP 2.1] ModelScan — Malicious Serialisation Check")

    try:
        result = subprocess.run(
            ["modelscan", "--path", str(MODEL_PATH), "--output", "json"],
            capture_output=True, text=True, timeout=60,
        )
        output = result.stdout.strip()
        if output:
            try:
                report = json.loads(output)
            except json.JSONDecodeError:
                report = {"raw_output": output, "issues_found": result.returncode != 0}
        else:
            report = {"issues_found": result.returncode != 0, "raw_output": result.stderr}

        issues = report.get("issues_found", False)
        total  = report.get("summary", {}).get("total_issues", None)
        if total is not None:
            issues = total > 0

        status = "[ALERT] Issues found!" if issues else "[OK] No malicious serialisation."
        print(f"  {status}")
        return report

    except FileNotFoundError:
        print("  [INFO] modelscan CLI not found — running manual pickle scan.")
        return manual_pickle_scan()
    except subprocess.TimeoutExpired:
        print("  [WARN] modelscan timed out.")
        return {"issues_found": False, "note": "timeout"}


def manual_pickle_scan() -> dict:
    """Scan for dangerous bytes in the model file."""
    DANGEROUS = [b"__reduce__", b"subprocess", b"os.system",
                 b"exec(", b"eval(", b"__import__"]
    issues = []
    try:
        data = MODEL_PATH.read_bytes()
        for pattern in DANGEROUS:
            if pattern in data:
                issues.append(pattern.decode(errors="replace"))
    except Exception as exc:
        return {"issues_found": True, "error": str(exc)}

    if issues:
        print(f"  [ALERT] Dangerous patterns found: {issues}")
    else:
        print("  [OK] No dangerous pickle patterns found.")
    return {"issues_found": len(issues) > 0, "dangerous_patterns": issues}

# ---------------------------------------------------------------------------
# 2. Weight norm analysis (fast heuristic)
# ---------------------------------------------------------------------------

def weight_norm_analysis() -> dict:
    """
    Check whether any output class has a suspiciously small weight norm.
    The triage.py backdoor script suppresses class-0 weights to 5% of normal.
    """
    print("\n[STEP 2.2] Weight Norm Analysis — Output Layer Heuristic")
    try:
        import torch
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

        # Find the output (last linear) layer weights
        out_key = None
        for key in state:
            if "out.weight" in key or ("fc" in key and "weight" in key):
                out_key = key

        if out_key is None:
            print("  [WARN] Could not identify output layer — skipping norm analysis.")
            return {}

        weights = state[out_key].numpy()   # shape: (n_classes, hidden_dim)
        norms   = np.linalg.norm(weights, axis=1)
        mean_norm = norms.mean()

        print(f"  Output layer key  : {out_key}")
        print(f"  Per-class norms   : " + "  ".join(f"cls{i}={n:.4f}" for i, n in enumerate(norms)))
        print(f"  Mean norm         : {mean_norm:.4f}")

        # Flag if any class norm is < 20% of mean
        suspicious = []
        for i, n in enumerate(norms):
            if n < mean_norm * 0.20:
                suspicious.append(i)
                print(f"  [ALERT] Class {i} weight norm ({n:.4f}) is only "
                      f"{n/mean_norm:.1%} of mean — possible backdoor!")

        if not suspicious:
            print("  [OK] All class norms within normal range.")

        return {"norms": norms.tolist(), "suspicious_classes": suspicious}

    except ImportError:
        print("  [WARN] torch not installed — skipping weight norm analysis.")
        return {}
    except Exception as exc:
        print(f"  [WARN] Weight norm analysis failed: {exc}")
        return {}

# ---------------------------------------------------------------------------
# 3. Simplified Neural Cleanse
# ---------------------------------------------------------------------------

def neural_cleanse_check() -> dict:
    """
    Approximate Neural Cleanse: reverse-engineer the minimal trigger for each
    output class and compute the anomaly index.
    High anomaly index → one class has a much smaller trigger → backdoor.
    """
    print("\n[STEP 2.3] Simplified Neural Cleanse — Backdoor Detection")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # Reconstruct DosageNet architecture from triage.py
        class DosageNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1  = nn.Linear(8, 64)
                self.relu = nn.ReLU()
                self.fc2  = nn.Linear(64, 32)
                self.out  = nn.Linear(32, 2)
            def forward(self, x):
                return self.out(self.relu(self.fc2(self.relu(self.fc1(x)))))

        model = DosageNet()
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        n_classes  = 2
        in_dim     = 8
        n_inputs   = 200
        n_steps    = 400

        norms = []
        X = torch.randn(n_inputs, in_dim)

        for cls_idx in range(n_classes):
            delta = torch.zeros(in_dim, requires_grad=True)
            opt   = torch.optim.Adam([delta], lr=0.01)
            target = torch.tensor([cls_idx] * n_inputs)

            for _ in range(n_steps):
                opt.zero_grad()
                logits = model(X + delta.unsqueeze(0))
                loss   = F.cross_entropy(logits, target)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    delta.clamp_(-1.0, 1.0)

            with torch.no_grad():
                norm = delta.abs().sum().item()
            norms.append(norm)
            print(f"  Class {cls_idx}: reverse-trigger L1 norm = {norm:.4f}")

        # Anomaly index
        norms_arr = np.array(norms)
        median    = np.median(norms_arr)
        mad       = np.median(np.abs(norms_arr - median))
        ai        = float((median - norms_arr.min()) / (mad + 1e-9))

        print(f"  Anomaly Index: {ai:.4f}  (threshold = {NC_THRESHOLD})")

        backdoor_detected = ai > NC_THRESHOLD
        verdict = "BACKDOOR_SUSPECTED" if backdoor_detected else "CLEAN"

        if backdoor_detected:
            suspicious_class = int(np.argmin(norms_arr))
            print(f"  [ALERT] Neural Cleanse flagged class {suspicious_class} as suspicious!")
        else:
            print("  [OK] Anomaly index within normal range.")

        return {
            "norms": norms,
            "anomaly_index": ai,
            "threshold": NC_THRESHOLD,
            "verdict": verdict,
        }

    except ImportError:
        print("  [WARN] torch not installed — skipping Neural Cleanse.")
        return {"verdict": "SKIPPED"}
    except Exception as exc:
        print(f"  [WARN] Neural Cleanse failed: {exc}")
        return {"verdict": "ERROR", "error": str(exc)}

# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*60}")
    print(f"STEP 2 — MODEL SCAN  [{now}]")
    print(f"{'='*60}")

    if not MODEL_PATH.exists():
        print(f"[ERROR] {MODEL_PATH} not found. Run triage/triage.py first.")
        sys.exit(1)

    modelscan_result = run_modelscan()
    norm_result      = weight_norm_analysis()
    nc_result        = neural_cleanse_check()

    print(f"\n{'='*60}")
    print("STEP 2 FINDINGS:")

    findings = []
    if modelscan_result.get("issues_found"):
        findings.append("Malicious serialisation detected by ModelScan.")

    if norm_result.get("suspicious_classes"):
        classes = norm_result["suspicious_classes"]
        findings.append(f"Output layer class(es) {classes} have suppressed weight norms.")

    if nc_result.get("verdict") == "BACKDOOR_SUSPECTED":
        ai = nc_result.get("anomaly_index", 0)
        findings.append(f"Neural Cleanse anomaly index {ai:.2f} exceeds threshold {NC_THRESHOLD}.")

    if findings:
        print("  [CONFIRMED] Vector 2: Backdoor Trigger in Fine-Tuned Model")
        for f in findings:
            print(f"    - {f}")
    else:
        print("  [NOT CONFIRMED] No backdoor evidence found.")

    print(f"{'='*60}")
    print("Next step: python triage/step3_supply_chain.py")


if __name__ == "__main__":
    main()
