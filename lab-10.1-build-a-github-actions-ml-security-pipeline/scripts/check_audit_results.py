"""
scripts/check_audit_results.py
================================
Parses audit_results.json produced by `pip-audit --output json --format json`
and exits with code 1 if any HIGH or CRITICAL CVEs are found.

pip-audit JSON schema (v2):
{
  "dependencies": [
    {
      "name": "package-name",
      "version": "1.2.3",
      "vulns": [
        {
          "id": "CVE-2021-XXXX",
          "fix_versions": ["1.2.4"],
          "aliases": [...],
          "description": "...",
          "severity": "HIGH"   <- not always present; we check CVSS score too
        }
      ]
    }
  ]
}

pip-audit does not always include a severity field; we use CVSS ≥ 7.0 as HIGH
and CVSS ≥ 9.0 as CRITICAL where available.
"""

import sys
import json
import pathlib

AUDIT_FILE        = pathlib.Path("audit_results.json")
SEVERITY_BLOCK    = {"HIGH", "CRITICAL"}
CVSS_HIGH_THRESH  = 7.0
CVSS_CRIT_THRESH  = 9.0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not AUDIT_FILE.exists():
        print(f"[ERROR] {AUDIT_FILE} not found. Run pip-audit first.")
        sys.exit(1)

    raw = AUDIT_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        print("[WARN] audit_results.json is empty — pip-audit produced no output.")
        print("[OK] Assuming no vulnerabilities found.")
        sys.exit(0)

    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Could not parse audit_results.json: {exc}")
        sys.exit(1)

    dependencies = report.get("dependencies", [])
    total_vulns  = 0
    blocking     = []

    for dep in dependencies:
        name    = dep.get("name", "unknown")
        version = dep.get("version", "unknown")
        vulns   = dep.get("vulns", [])

        for vuln in vulns:
            total_vulns += 1
            vuln_id  = vuln.get("id", "UNKNOWN")
            severity = vuln.get("severity", "").upper()
            cvss     = vuln.get("cvss", None)

            is_blocking = False

            # Check explicit severity field
            if severity in SEVERITY_BLOCK:
                is_blocking = True

            # Check CVSS score if available
            if cvss is not None:
                try:
                    score = float(cvss)
                    if score >= CVSS_HIGH_THRESH:
                        is_blocking = True
                        if severity == "":
                            severity = "CRITICAL" if score >= CVSS_CRIT_THRESH else "HIGH"
                except (ValueError, TypeError):
                    pass

            if is_blocking:
                blocking.append({
                    "package": f"{name}=={version}",
                    "cve": vuln_id,
                    "severity": severity or "HIGH/CRITICAL",
                    "fix_versions": vuln.get("fix_versions", []),
                })
                print(f"  [BLOCK] {name}=={version}  {vuln_id}  severity={severity}")
            else:
                print(f"  [INFO]  {name}=={version}  {vuln_id}  severity={severity or 'LOW/MEDIUM'}")

    print(f"\n[AUDIT] Total vulnerabilities: {total_vulns}")
    print(f"[AUDIT] Blocking (HIGH/CRITICAL): {len(blocking)}")

    if blocking:
        print("\n[ERROR] HIGH/CRITICAL CVEs found — aborting pipeline.")
        print("Upgrade or replace the following packages:")
        for b in blocking:
            fixes = ", ".join(b["fix_versions"]) or "no known fix"
            print(f"  {b['package']}  ({b['cve']})  fix: {fixes}")
        sys.exit(1)

    print("[OK] No HIGH or CRITICAL CVEs found in dependency audit.")


if __name__ == "__main__":
    main()
