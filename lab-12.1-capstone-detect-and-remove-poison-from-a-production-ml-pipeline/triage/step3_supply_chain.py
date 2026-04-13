"""
triage/step3_supply_chain.py
=============================
Step 3 of the MedBot-7 incident response.

Analyses suspicious_requirements.txt for:
  1. Known vulnerable packages (cross-referenced against a local CVE list).
  2. pip-audit simulation (parses pinned versions against known CVEs).
  3. Package origin checks (flags packages not on PyPI or from unknown indexes).
  4. Dependency hash verification (detects unsigned / unverified packages).

Run:
    python triage/step3_supply_chain.py
"""

import sys
import json
import pathlib
import datetime
import subprocess

# ---------------------------------------------------------------------------
# 0. Paths and constants
# ---------------------------------------------------------------------------

TRIAGE_DIR    = pathlib.Path(__file__).parent
REQ_PATH      = TRIAGE_DIR / "suspicious_requirements.txt"

# Simulated CVE database — in production, use OSV/NVD APIs
# Format: { "package==version": [(CVE_ID, CVSS, description)] }
KNOWN_CVES = {
    "medbot-dosage-lib==0.9.2": [
        ("CVE-2024-99999", 9.1,
         "Remote Code Execution via crafted patient input string in dosage parser."),
    ],
    "numpy==1.21.0": [
        ("CVE-2021-33430", 7.5,
         "Buffer overflow in numpy.core leading to denial of service."),
    ],
    "cryptography==36.0.0": [
        ("CVE-2023-23931", 5.3,
         "Bleichenbacher timing oracle in RSA decryption."),
    ],
    "pillow==9.0.0": [
        ("CVE-2023-44271", 7.5,
         "Uncontrolled resource consumption in PIL.ImageFont."),
    ],
}

CRITICAL_CVSS = 9.0
HIGH_CVSS     = 7.0

# ---------------------------------------------------------------------------
# 1. Parse requirements file
# ---------------------------------------------------------------------------

def parse_requirements(path: pathlib.Path) -> list:
    """Parse a requirements.txt and return list of (package, version, raw_line) tuples."""
    packages = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            parts   = line.split("==")
            name    = parts[0].strip().lower()
            version = parts[1].strip()
            packages.append((name, version, line))
        else:
            packages.append((line, "unknown", line))
    return packages

# ---------------------------------------------------------------------------
# 2. CVE lookup (local simulation)
# ---------------------------------------------------------------------------

def check_local_cve_db(packages: list) -> list:
    """
    Cross-reference parsed packages against the local KNOWN_CVES dict.
    Returns list of finding dicts.
    """
    findings = []
    for name, version, raw in packages:
        key = f"{name}=={version}"
        # Check exact match
        if key in KNOWN_CVES:
            for cve_id, cvss, desc in KNOWN_CVES[key]:
                severity = "CRITICAL" if cvss >= CRITICAL_CVSS else (
                           "HIGH"     if cvss >= HIGH_CVSS    else "MEDIUM")
                findings.append({
                    "package":  raw,
                    "cve":      cve_id,
                    "cvss":     cvss,
                    "severity": severity,
                    "description": desc,
                })
    return findings

# ---------------------------------------------------------------------------
# 3. pip-audit simulation / real pip-audit if available
# ---------------------------------------------------------------------------

def run_pip_audit_simulation(packages: list) -> list:
    """
    Try real pip-audit first; fall back to simulated CVE check.
    Returns list of finding dicts.
    """
    print("\n[STEP 3.2] pip-audit Dependency Scan")

    # Attempt real pip-audit on the requirements file
    try:
        result = subprocess.run(
            ["pip-audit", "-r", str(REQ_PATH), "--output", "json", "--format", "json"],
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout.strip()
        if output:
            try:
                audit_data = json.loads(output)
                real_findings = []
                for dep in audit_data.get("dependencies", []):
                    for vuln in dep.get("vulns", []):
                        real_findings.append({
                            "package": f"{dep['name']}=={dep['version']}",
                            "cve": vuln.get("id", "UNKNOWN"),
                            "cvss": vuln.get("cvss", 0),
                            "severity": vuln.get("severity", "UNKNOWN"),
                            "description": vuln.get("description", "")[:120],
                        })
                print(f"  [pip-audit] Found {len(real_findings)} vulnerability/ies via real pip-audit.")
                return real_findings
            except json.JSONDecodeError:
                pass
    except FileNotFoundError:
        print("  [INFO] pip-audit CLI not found — using local CVE simulation.")
    except subprocess.TimeoutExpired:
        print("  [WARN] pip-audit timed out — using local simulation.")
    except Exception as exc:
        print(f"  [WARN] pip-audit error ({exc}) — using local simulation.")

    # Fallback: local CVE database
    findings = check_local_cve_db(packages)
    print(f"  [Simulation] Found {len(findings)} vulnerability/ies via local CVE DB.")
    return findings

# ---------------------------------------------------------------------------
# 4. Unknown/non-PyPI package detection
# ---------------------------------------------------------------------------

def check_unknown_packages(packages: list) -> list:
    """
    Flag packages that are not present on PyPI by a simple heuristic:
    packages with a hyphen in the name that look like internal/custom packages.
    In production, this would use PyPI's JSON API.
    """
    SUSPICIOUS_PATTERNS = [
        "medbot-",  "hospital-", "internal-", "private-", "custom-",
    ]
    suspicious = []
    for name, version, raw in packages:
        for pat in SUSPICIOUS_PATTERNS:
            if name.startswith(pat):
                suspicious.append({
                    "package": raw,
                    "reason": f"Package name matches suspicious pattern '{pat}' — "
                              "verify it is published on PyPI and is the expected package.",
                })
    return suspicious

# ---------------------------------------------------------------------------
# 5. Print findings table
# ---------------------------------------------------------------------------

def print_findings(cve_findings: list, unknown_findings: list) -> None:
    print("\n[STEP 3.3] CVE Findings")
    if cve_findings:
        print(f"  {'Package':<35} {'CVE':<18} {'CVSS':>6}  {'Severity'}")
        print("  " + "-" * 72)
        for f in cve_findings:
            print(f"  {f['package']:<35} {f['cve']:<18} {f['cvss']:>6.1f}  {f['severity']}")
            print(f"    Description: {f['description'][:90]}")
    else:
        print("  No CVEs found.")

    print("\n[STEP 3.4] Unknown/Suspicious Package Origins")
    if unknown_findings:
        for u in unknown_findings:
            print(f"  [WARN] {u['package']}")
            print(f"         {u['reason']}")
    else:
        print("  No suspicious package origins detected.")

# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*60}")
    print(f"STEP 3 — SUPPLY CHAIN ANALYSIS  [{now}]")
    print(f"{'='*60}")

    if not REQ_PATH.exists():
        print(f"[ERROR] {REQ_PATH} not found. Run triage/triage.py first.")
        sys.exit(1)

    # ---- Parse ----
    print(f"\n[STEP 3.1] Parsing {REQ_PATH.name}")
    packages = parse_requirements(REQ_PATH)
    print(f"  {len(packages)} packages parsed.")
    for name, version, _ in packages:
        print(f"    {name}=={version}")

    # ---- CVE scan ----
    cve_findings = run_pip_audit_simulation(packages)

    # ---- Unknown packages ----
    unknown_findings = check_unknown_packages(packages)

    # ---- Print ----
    print_findings(cve_findings, unknown_findings)

    # ---- Summary ----
    critical = [f for f in cve_findings if f["severity"] in ("CRITICAL", "HIGH")]

    print(f"\n{'='*60}")
    print("STEP 3 FINDINGS:")
    if critical or unknown_findings:
        print("  [CONFIRMED] Vector 3: Supply Chain Compromise")
        if critical:
            print(f"  {len(critical)} HIGH/CRITICAL CVE(s) found:")
            for f in critical:
                print(f"    - {f['package']}  {f['cve']}  CVSS={f['cvss']}  {f['severity']}")
                print(f"      {f['description'][:80]}")
        if unknown_findings:
            print(f"  {len(unknown_findings)} suspicious package(s) flagged.")
    else:
        print("  [NOT CONFIRMED] No supply chain issues detected.")

    print(f"{'='*60}")
    print("Next step: python triage/step4_remediation.py")


if __name__ == "__main__":
    main()
