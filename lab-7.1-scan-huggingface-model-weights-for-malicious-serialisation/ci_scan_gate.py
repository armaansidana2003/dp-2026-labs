"""
ci_scan_gate.py — Reusable CI/CD Model Scanning Gate
Data Poisoning Protection Course — Lab 7.1

PURPOSE
-------
This script is designed to be called as a step inside a CI/CD pipeline
(GitHub Actions, GitLab CI, Jenkins, etc.) to prevent malicious model files
from passing through a deployment gate.

USAGE
-----
From the command line:
    python ci_scan_gate.py path/to/model.pt
    python ci_scan_gate.py path/to/model.pkl

From a GitHub Actions workflow step:
    - name: Scan model artifact
      run: python ci_scan_gate.py ${{ env.MODEL_PATH }}

Exit codes:
    0  — scan passed, model is safe to use
    1  — scan failed, CRITICAL or HIGH findings detected  (pipeline should abort)
    2  — invocation error (missing path, file not found)

From Python:
    from ci_scan_gate import scan_model_file
    result = scan_model_file("model.pt")
    # result: {"safe": True, "findings": [], "exit_code": 0}
"""

import json
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core scan function
# ---------------------------------------------------------------------------

def scan_model_file(path: str) -> dict:
    """
    Run ModelScan against a model file and return a structured result.

    Parameters
    ----------
    path : str
        Absolute or relative path to the model file to scan.

    Returns
    -------
    dict with keys:
        safe       : bool   — True if no CRITICAL/HIGH findings
        findings   : list   — list of finding dicts (severity, description, source)
        exit_code  : int    — 0 if safe, 1 if unsafe, 2 on invocation error
        raw_output : str    — raw stdout from modelscan for logging

    Side-effects
    ------------
    Prints a summary to stdout.
    Calls sys.exit(1) if unsafe (making it suitable for direct CI/CD usage).
    Does NOT call sys.exit when imported as a module — caller controls flow.
    """
    path = str(path)

    # Validate the file exists before shelling out
    if not Path(path).exists():
        print(f"[ERROR] File not found: {path}")
        return {"safe": False, "findings": [], "exit_code": 2, "raw_output": ""}

    print(f"[ModelScan Gate] Scanning: {path}")

    # ---------------------------------------------------------------------------
    # Attempt 1 — JSON output mode (ModelScan >= 0.5.x)
    # ---------------------------------------------------------------------------
    findings    = []
    raw_output  = ""
    json_parsed = False

    try:
        proc = subprocess.run(
            ["modelscan", "--json", "-p", path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        raw_output = proc.stdout.strip()

        if raw_output.startswith("{") or raw_output.startswith("["):
            data = json.loads(raw_output)
            if isinstance(data, dict):
                issues = data.get("issues", data.get("findings", []))
            elif isinstance(data, list):
                issues = data
            else:
                issues = []

            for issue in issues:
                findings.append({
                    "severity":    issue.get("severity", "UNKNOWN").upper(),
                    "description": issue.get("description", str(issue)),
                    "source":      issue.get("source", path),
                })
            json_parsed = True

    except FileNotFoundError:
        print("[ERROR] 'modelscan' not found. Install with: pip install modelscan")
        return {"safe": False, "findings": [], "exit_code": 2, "raw_output": ""}
    except json.JSONDecodeError:
        # JSON parse failed — fall through to plain-text mode
        json_parsed = False
    except subprocess.TimeoutExpired:
        print(f"[ERROR] ModelScan timed out scanning: {path}")
        return {"safe": False, "findings": [], "exit_code": 2, "raw_output": ""}

    # ---------------------------------------------------------------------------
    # Attempt 2 — Plain-text mode (older ModelScan versions or JSON unavailable)
    # ---------------------------------------------------------------------------
    if not json_parsed:
        try:
            proc = subprocess.run(
                ["modelscan", "-p", path],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            print("[ERROR] 'modelscan' not found. Install with: pip install modelscan")
            return {"safe": False, "findings": [], "exit_code": 2, "raw_output": ""}
        except subprocess.TimeoutExpired:
            print(f"[ERROR] ModelScan timed out scanning: {path}")
            return {"safe": False, "findings": [], "exit_code": 2, "raw_output": ""}

        raw_output = (proc.stdout + proc.stderr).strip()

        # Heuristic: look for severity keywords in each line
        for line in raw_output.splitlines():
            upper_line = line.upper()
            for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
                if sev in upper_line:
                    findings.append({
                        "severity":    sev,
                        "description": line.strip(),
                        "source":      path,
                    })
                    break

        # Explicit safe markers
        if ("no issues" in raw_output.lower()
                or "no dangerous" in raw_output.lower()
                or "no findings" in raw_output.lower()):
            findings = []

    # ---------------------------------------------------------------------------
    # Evaluate results
    # ---------------------------------------------------------------------------
    BLOCKING_SEVERITIES = {"CRITICAL", "HIGH"}

    blocking = [f for f in findings if f["severity"] in BLOCKING_SEVERITIES]
    is_safe  = len(blocking) == 0

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    _print_gate_summary(path, is_safe, findings)

    exit_code = 0 if is_safe else 1
    return {
        "safe":       is_safe,
        "findings":   findings,
        "exit_code":  exit_code,
        "raw_output": raw_output,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_gate_summary(path: str, is_safe: bool, findings: list) -> None:
    """Print a formatted CI/CD-style gate summary to stdout."""
    sep = "-" * 55

    print(sep)
    if is_safe:
        print(f"  [PASS] Model scan passed")
        print(f"  File  : {path}")
        if findings:
            # Informational (LOW/MEDIUM) findings — not blocking
            print(f"  Info  : {len(findings)} non-blocking finding(s) noted")
            for f in findings:
                print(f"    [{f['severity']}] {f['description']}")
        else:
            print(f"  Result: No issues found")
    else:
        print(f"  [FAIL] Model scan FAILED — unsafe model detected")
        print(f"  File  : {path}")
        print(f"  Blocking findings ({len([f for f in findings if f['severity'] in {'CRITICAL','HIGH'}])}):")
        for f in findings:
            if f["severity"] in {"CRITICAL", "HIGH"}:
                print(f"    [{f['severity']}] {f['description']}")
                print(f"             Source: {f['source']}")
    print(sep)


# ---------------------------------------------------------------------------
# Main — called directly from CI/CD pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Entry point for CI/CD pipeline usage.

    Reads the model path from sys.argv[1], scans it, and exits with:
      0  — safe
      1  — unsafe (pipeline should fail the build/deployment)
      2  — invocation error
    """
    if len(sys.argv) < 2:
        print("Usage: python ci_scan_gate.py <path-to-model-file>")
        print("")
        print("Examples:")
        print("  python ci_scan_gate.py model.pt")
        print("  python ci_scan_gate.py model.pkl")
        print("  python ci_scan_gate.py model.safetensors")
        sys.exit(2)

    model_path = sys.argv[1]
    result     = scan_model_file(model_path)

    if not result["safe"]:
        print(
            "\n[CI/CD] Halting pipeline. The model artifact did not pass the security gate.\n"
            "        Review the findings above and either:\n"
            "          a) Source a trusted replacement model, OR\n"
            "          b) Escalate to the security team for investigation.\n"
        )
        sys.exit(result["exit_code"])

    # Safe — pipeline may continue
    print("\n[CI/CD] Gate passed. Continuing pipeline.\n")
    sys.exit(0)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# GitHub Actions usage example (commented out — for reference only)
# ---------------------------------------------------------------------------
#
# Add this to your .github/workflows/ml_deploy.yml:
#
# jobs:
#   security-gate:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#
#       - name: Set up Python
#         uses: actions/setup-python@v5
#         with:
#           python-version: "3.11"
#
#       - name: Install dependencies
#         run: pip install modelscan
#
#       - name: Download model artifact
#         run: |
#           aws s3 cp s3://my-model-bucket/model.pt ./model.pt
#           # or: huggingface-cli download org/repo model.pt
#
#       - name: Security gate — scan model before deploy
#         run: python ci_scan_gate.py ./model.pt
#         # If this step exits with code 1, the workflow fails here
#         # and the model is never deployed.
#
#       - name: Deploy model
#         run: ./deploy.sh  # only reached if scan passed
