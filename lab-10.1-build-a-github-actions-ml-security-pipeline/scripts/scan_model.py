"""
scripts/scan_model.py
=====================
Runs ModelScan on model.pt to detect malicious serialisation payloads
(e.g., pickle exploits embedded in PyTorch checkpoints).
Exits with code 1 if any issues are found (CI gate).

ModelScan docs: https://github.com/protectai/model-scan
"""

import sys
import json
import pathlib
import subprocess

MODEL_PATH  = pathlib.Path("model.pt")
REPORT_PATH = pathlib.Path("modelscan_report.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_modelscan_cli(model_path: pathlib.Path) -> dict:
    """
    Invoke the modelscan CLI via subprocess and return parsed JSON report.
    Falls back to a manual pickle inspection if modelscan is not available.
    """
    try:
        result = subprocess.run(
            ["modelscan", "--path", str(model_path), "--output", "json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        # modelscan exits 0 if clean, non-zero if issues found
        output = result.stdout.strip()
        if output:
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                # Wrap plain-text output in a dict
                return {
                    "raw_output": output,
                    "issues_found": result.returncode != 0,
                }
        return {"issues_found": result.returncode != 0, "raw_output": result.stderr}

    except FileNotFoundError:
        print("[WARN] modelscan CLI not found — running manual pickle inspection.")
        return manual_pickle_inspect(model_path)
    except subprocess.TimeoutExpired:
        print("[ERROR] modelscan timed out.")
        return {"issues_found": True, "error": "timeout"}


def manual_pickle_inspect(model_path: pathlib.Path) -> dict:
    """
    Minimal safety check: open the file and look for known dangerous pickle
    opcodes that indicate arbitrary code execution payloads.
    This is NOT a substitute for ModelScan — it is a fallback gate.
    """
    DANGEROUS_PATTERNS = [
        b"__reduce__",
        b"subprocess",
        b"os.system",
        b"exec(",
        b"eval(",
        b"importlib",
        b"__import__",
    ]

    issues = []
    try:
        data = model_path.read_bytes()
        for pattern in DANGEROUS_PATTERNS:
            if pattern in data:
                issues.append(f"Dangerous pattern found: {pattern.decode(errors='replace')}")
    except Exception as exc:
        return {"issues_found": True, "error": str(exc)}

    return {
        "issues_found": len(issues) > 0,
        "issues": issues,
        "method": "manual_pickle_inspect",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] {MODEL_PATH} not found. Create or download the model artifact first.")
        sys.exit(1)

    print(f"[INFO] Scanning {MODEL_PATH} with ModelScan …")
    report = run_modelscan_cli(MODEL_PATH)

    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"[INFO] Scan report saved → {REPORT_PATH}")

    issues_found = report.get("issues_found", False)

    # ModelScan's JSON structure uses "summary.total_issues"
    if isinstance(report, dict):
        summary = report.get("summary", {})
        total_issues = summary.get("total_issues", None)
        if total_issues is not None:
            issues_found = total_issues > 0

    if issues_found:
        print("[ERROR] ModelScan found issues in the model artifact — aborting pipeline.")
        if "issues" in report:
            for issue in report["issues"]:
                print(f"  - {issue}")
        sys.exit(1)

    print("[OK] No malicious serialisation found in model artifact.")


if __name__ == "__main__":
    main()
