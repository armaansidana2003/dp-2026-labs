"""
Lab 7.2 — Dependency Auditing & Artifact Signature Verification
Data Poisoning Protection Course

This script walks through four steps:
  1. Run pip-audit programmatically — parse CVE table
  2. Generate hash-pinned requirements.txt
  3. Sign a model artifact (Sigstore simulation with HMAC)
  4. Tamper detection — verify_artifact() catches byte-level modification

Run with:  python lab_7_2_dependency_signing.py

Dependencies:  pip install -r requirements.txt
               (hashlib, hmac, json, subprocess are stdlib — no install needed)
"""

import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class TamperDetectedError(Exception):
    """Raised when a model artifact's hash does not match its signing bundle."""
    pass


# ===========================================================================
# STEP 1 — Automated dependency vulnerability scanning with pip-audit
# ===========================================================================

def step1_run_pip_audit() -> list:
    """
    Run pip-audit against the current Python environment and display a
    formatted vulnerability table.

    pip-audit queries the Python Packaging Advisory Database (PyPA Advisory
    DB) and the OSV database for known CVEs in every installed package.

    We call it via subprocess with --output json so we can parse the results
    programmatically rather than scraping plain-text output.

    Returns a list of vulnerability dicts found (empty list means clean).
    """
    print("\n" + "#" * 60)
    print("# STEP 1 — Dependency Vulnerability Scanning (pip-audit)")
    print("#" * 60)
    print("\n[*] Running pip-audit against the current Python environment...")
    print(    "    This queries the PyPA Advisory DB and OSV for known CVEs.")
    print(    "    Command: pip-audit --output json\n")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--output", "json", "--progress-spinner", "off"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        raw = result.stdout.strip()
    except FileNotFoundError:
        # Try the console script entry point directly
        try:
            result = subprocess.run(
                ["pip-audit", "--output", "json", "--progress-spinner", "off"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            raw = result.stdout.strip()
        except FileNotFoundError:
            print("[ERROR] pip-audit not found. Install with: pip install pip-audit")
            print("[INFO]  Continuing lab with simulated vulnerability data.\n")
            return _simulate_pip_audit_results()
    except subprocess.TimeoutExpired:
        print("[ERROR] pip-audit timed out. Using simulated results.")
        return _simulate_pip_audit_results()

    # ---------------------------------------------------------------------------
    # Parse JSON output
    # pip-audit JSON schema:
    #   { "dependencies": [ { "name": str, "version": str, "vulns": [...] } ] }
    # Each vuln: { "id": str, "fix_versions": [...], "aliases": [...], ... }
    # ---------------------------------------------------------------------------
    vulnerabilities = []

    if not raw:
        print("[INFO] pip-audit produced no output. Using simulated results.")
        return _simulate_pip_audit_results()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[WARN] Could not parse pip-audit JSON output. Raw output:\n{raw}")
        return _simulate_pip_audit_results()

    # The top-level key may be "dependencies" or the root may be a list
    if isinstance(data, dict):
        deps = data.get("dependencies", [])
    elif isinstance(data, list):
        deps = data
    else:
        deps = []

    for dep in deps:
        pkg_name    = dep.get("name", "unknown")
        pkg_version = dep.get("version", "?")
        vulns       = dep.get("vulns", [])

        for vuln in vulns:
            vuln_id      = vuln.get("id", "UNKNOWN")
            fix_versions = vuln.get("fix_versions", [])
            description  = vuln.get("description", "No description available.")
            aliases      = vuln.get("aliases", [])

            # pip-audit does not always include severity — infer from CVE ID prefix
            severity = _infer_severity(vuln_id, aliases)

            vulnerabilities.append({
                "package":      pkg_name,
                "version":      pkg_version,
                "vuln_id":      vuln_id,
                "aliases":      aliases,
                "severity":     severity,
                "fix_versions": fix_versions,
                "description":  description[:120] + "..." if len(description) > 120 else description,
            })

    _print_vulnerability_table(vulnerabilities)
    return vulnerabilities


def _infer_severity(vuln_id: str, aliases: list) -> str:
    """
    Infer severity from the vulnerability ID or aliases.
    pip-audit does not always include CVSS severity in its JSON output.
    We look for GHSA or CVE prefixes and label HIGH if present as a safe default.
    """
    all_ids = [vuln_id] + aliases
    for vid in all_ids:
        vid_upper = vid.upper()
        if vid_upper.startswith("CVE") or vid_upper.startswith("GHSA"):
            return "HIGH"   # Conservative default — treat known CVEs as HIGH
    return "MEDIUM"


def _print_vulnerability_table(vulns: list) -> None:
    """Print a formatted vulnerability report to stdout."""
    if not vulns:
        print("[+] No vulnerabilities found in the current environment.")
        print("    Your dependencies are clean according to the PyPA Advisory DB.\n")
        return

    print(f"[!] Found {len(vulns)} vulnerability/vulnerabilities:\n")

    # Header
    header = f"{'Package':<20} {'Version':<12} {'CVE/ID':<20} {'Severity':<10} {'Fix Available':<15}"
    print(header)
    print("-" * len(header))

    # Rows
    high_critical = []
    for v in vulns:
        fix = v["fix_versions"][0] if v["fix_versions"] else "None known"
        row = (
            f"{v['package']:<20} {v['version']:<12} "
            f"{v['vuln_id']:<20} {v['severity']:<10} {fix:<15}"
        )
        print(row)
        if v["severity"].upper() in ("CRITICAL", "HIGH"):
            high_critical.append(v)

    print()

    if high_critical:
        print(f"[WARNING] {len(high_critical)} HIGH or CRITICAL severity package(s) found:")
        for v in high_critical:
            fix = v["fix_versions"][0] if v["fix_versions"] else "none known"
            print(f"  -> {v['package']} {v['version']} ({v['vuln_id']}) — fix: {fix}")
        print("\n  Action required: update these packages before deploying.")
        print("  Run: pip install --upgrade <package_name>\n")
    else:
        print("[*] No CRITICAL or HIGH severity vulnerabilities. Review MEDIUM/LOW at your discretion.")


def _simulate_pip_audit_results() -> list:
    """
    Return simulated pip-audit results for environments where pip-audit
    is unavailable or times out.  Used for demonstration purposes only.
    """
    print("[SIM] Using simulated pip-audit output for demonstration.\n")
    simulated = [
        {
            "package":      "requests",
            "version":      "2.25.0",
            "vuln_id":      "GHSA-j8r2-6x86-q33q",
            "aliases":      ["CVE-2023-32681"],
            "severity":     "MEDIUM",
            "fix_versions": ["2.31.0"],
            "description":  "Requests forwards proxy-authorization header to destination servers on redirects.",
        },
        {
            "package":      "cryptography",
            "version":      "38.0.0",
            "vuln_id":      "GHSA-w7pp-m8wf-vj6r",
            "aliases":      ["CVE-2023-49083"],
            "severity":     "HIGH",
            "fix_versions": ["41.0.6"],
            "description":  "Null-pointer dereference in PKCS12 parsing — affects OpenSSL backend.",
        },
    ]
    _print_vulnerability_table(simulated)
    return simulated


# ===========================================================================
# STEP 2 — Hash-pinned requirements generation
# ===========================================================================

def step2_generate_hash_pinned_requirements(
    input_path:  str = "requirements.txt",
    output_path: str = "requirements_pinned.txt",
) -> str:
    """
    Read requirements.txt and generate a hash-pinned version.

    Standard pinning (torch==2.0.0) prevents version drift but not
    mirror-substitution attacks — a compromised PyPI mirror could serve a
    different wheel file with the same version string.

    Hash pinning ties pip to the exact file content.  pip will reject any
    wheel whose SHA-256 hash does not match what is declared in requirements.txt.
    The format is:

        torch==2.0.0 \\
            --hash=sha256:<hex>

    In production the real hashes come from pip download:
        pip download --no-deps -d wheels torch==2.0.0
        pip hash wheels/torch-*.whl

    For this lab we generate deterministic fake hashes (sha256 of the package
    name + version string) so the format is correct and replaceable.
    """
    print("\n" + "#" * 60)
    print("# STEP 2 — Generating Hash-Pinned Requirements")
    print("#" * 60)

    # --- Read the source requirements.txt ---
    req_path = Path(input_path)
    if not req_path.exists():
        # Fall back to a default list for standalone execution
        print(f"[WARN] {input_path} not found — using default package list.")
        lines = [
            "pip-audit>=2.7.0",
            "pip-tools>=7.3.0",
            "sigstore>=2.0.0",
            "torch>=2.0.0",
            "safetensors>=0.4.0",
        ]
    else:
        lines = req_path.read_text().splitlines()

    print(f"\n[*] Reading packages from: {input_path}")
    print(f"[*] Will write pinned requirements to: {output_path}\n")

    # --- Parse and generate pinned lines ---
    pinned_lines = [
        "# Hash-pinned requirements",
        "# Generated by lab_7_2_dependency_signing.py",
        "# Replace DEMO hashes with real hashes from: pip hash <wheel_file>",
        "# Usage: pip install --require-hashes -r requirements_pinned.txt",
        "#",
        "# WARNING: The SHA-256 hashes below are DEMO values (hash of pkg+version).",
        "# Replace them with real wheel hashes before using in production.",
        "#",
        "",
    ]

    processed_packages = []

    for raw_line in lines:
        line = raw_line.strip()

        # Skip blank lines and comments
        if not line or line.startswith("#"):
            pinned_lines.append(raw_line)
            continue

        # Parse package name and version specifier
        # Handles: torch>=2.0.0, torch==2.0.0, torch~=2.0.0, torch, torch[cuda]
        match = re.match(
            r"^([A-Za-z0-9_\-\[\]]+)"   # package name (with extras)
            r"\s*([><=!~^]+)?\s*"         # version operator (optional)
            r"([0-9][0-9a-zA-Z.\-*]*)?",  # version number (optional)
            line,
        )

        if not match:
            pinned_lines.append(f"# SKIPPED (could not parse): {raw_line}")
            continue

        pkg_name    = match.group(1)
        operator    = match.group(2) or "=="
        version_str = match.group(3) or "0.0.0"

        # Normalise: always use == for pinning (exact version)
        pinned_version = version_str

        # Generate a deterministic fake SHA-256 hash for demonstration
        # In production: replace with actual `pip hash` output
        fake_hash = _generate_demo_hash(pkg_name, pinned_version)

        pinned_line = (
            f"{pkg_name}=={pinned_version} \\\n"
            f"    --hash=sha256:{fake_hash}"
        )
        pinned_lines.append(pinned_line)

        processed_packages.append({
            "package": pkg_name,
            "version": pinned_version,
            "hash":    fake_hash,
        })

        print(f"  Pinned: {pkg_name}=={pinned_version}")
        print(f"          sha256:{fake_hash[:32]}...  [DEMO HASH]")

    # --- Write output file ---
    output_content = "\n".join(pinned_lines) + "\n"
    Path(output_path).write_text(output_content)

    print(f"\n[+] Written {len(processed_packages)} pinned package(s) to: {output_path}")
    print(    "\n[IMPORTANT] The hashes above are DEMONSTRATION values.")
    print(    "            Replace them with real hashes using:")
    print(    "              pip download --no-deps -d ./wheels <package>==<version>")
    print(    "              pip hash ./wheels/<wheel_file>.whl")
    print(    "            Then copy the sha256 hash into requirements_pinned.txt.\n")

    return output_path


def _generate_demo_hash(package_name: str, version: str) -> str:
    """
    Generate a deterministic but fake SHA-256 hash for a package+version.

    This is purely for demonstration — it produces a consistent 64-char hex
    string so the requirements file format is valid and parseable.

    NEVER use this in production. Always use real hashes from pip hash.
    """
    # We use HMAC with a fixed demo key so the hash is deterministic across runs
    demo_key = b"udege-lab-7.2-demo-not-a-real-hash"
    payload  = f"{package_name.lower()}=={version}".encode("utf-8")
    return hmac.new(demo_key, payload, digestmod=hashlib.sha256).hexdigest()


# ===========================================================================
# STEP 3 — Model artifact signing (Sigstore simulation with HMAC)
# ===========================================================================

def step3_sign_model_artifact(
    model_path:  str = "model_artifact.pt",
    bundle_path: str = "model_artifact.pt.bundle.json",
) -> dict:
    """
    Create a PyTorch model, compute its SHA-256, and produce a signing bundle.

    Real Sigstore keyless signing flow:
      1. Developer/CI authenticates to an OIDC provider (GitHub Actions, Google).
      2. Fulcio CA issues a short-lived X.509 certificate bound to the OIDC identity.
      3. The artifact hash is signed with the certificate's private key (ECDSA).
      4. The (certificate, signature) pair is submitted to the Rekor transparency log.
      5. Verifiers check the transparency log — they never need the signer's long-lived key.

    This lab simulates steps 3-5 with HMAC-SHA256 because:
      - OIDC requires an external identity provider (not available in a local lab).
      - The bundle JSON structure mirrors the real Sigstore bundle format.
      - The verify_artifact() logic (Step 4) is directly portable to a real implementation.

    The signing secret (SIGNING_SECRET) would be stored in a secrets manager
    (AWS Secrets Manager, HashiCorp Vault, GitHub Actions Secrets) in production.

    Returns the bundle dict that was written to bundle_path.
    """
    print("\n" + "#" * 60)
    print("# STEP 3 — Model Artifact Signing (Sigstore Simulation)")
    print("#" * 60)

    # --- Create a simple model ---
    print("\n[*] Creating model_artifact.pt ...")
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    model.eval()
    torch.save(model.state_dict(), model_path)
    file_size = Path(model_path).stat().st_size
    print(f"[+] model_artifact.pt created ({file_size} bytes)")
    print(    "    Architecture: Linear(64->32)->ReLU->Linear(32->16)->ReLU->Linear(16->4)")

    # --- Compute SHA-256 of the model file ---
    print("\n[*] Computing SHA-256 of model_artifact.pt ...")
    artifact_hash = _compute_file_sha256(model_path)
    print(f"[+] SHA-256: {artifact_hash}")

    # --- Simulate Sigstore signing with HMAC ---
    # In real Sigstore: the hash is signed with an ephemeral ECDSA key
    # issued by Fulcio CA.  Here we sign with HMAC-SHA256 using a
    # secret key.  The bundle JSON structure is the same.
    print("\n[*] Signing artifact hash (HMAC-SHA256 simulation) ...")

    # In production: retrieve this from AWS Secrets Manager / Vault / CI secrets
    # NEVER hardcode a real signing secret in source code.
    SIGNING_SECRET = b"lab-demo-signing-secret-replace-in-production"  # noqa: N806

    signature = hmac.new(
        SIGNING_SECRET,
        artifact_hash.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    # --- Build the signing bundle ---
    # This mirrors the structure of a real Sigstore bundle (.sigstore JSON)
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    bundle = {
        "_comment": (
            "DEMO bundle — simulates Sigstore keyless signing. "
            "In production use: sigstore sign --oidc-issuer <issuer> model_artifact.pt"
        ),
        "mediaType":     "application/vnd.dev.sigstore.bundle+json;version=0.1",
        "artifactHash":  {
            "algorithm": "sha2-256",
            "digest":    artifact_hash,
        },
        "signature":     {
            "algorithm": "hmac-sha256",
            "value":     signature,
            "note":      "In real Sigstore this is an ECDSA signature over the artifact hash",
        },
        "verificationMaterial": {
            "signerIdentity": "ci@udege-lab.internal",
            "issuer":         "https://accounts.google.com",   # simulated OIDC issuer
            "transparencyLog": {
                "logName":   "rekor.sigstore.dev",
                "logIndex":  "SIMULATED",
                "entryHash": hashlib.sha256(signature.encode()).hexdigest(),
            },
        },
        "metadata": {
            "createdAt": timestamp,
            "modelFile": str(Path(model_path).name),
            "labNote":   "Replace SIGNING_SECRET with a real Sigstore workflow in production",
        },
    }

    # Write the bundle to disk
    bundle_json = json.dumps(bundle, indent=2)
    Path(bundle_path).write_text(bundle_json)

    print(f"[+] Signing bundle written to: {bundle_path}")
    print(f"    Artifact hash : {artifact_hash[:32]}...")
    print(f"    HMAC signature: {signature[:32]}...")
    print(f"    Timestamp     : {timestamp}")
    print(f"    Signer        : {bundle['verificationMaterial']['signerIdentity']}")

    print("\n[INFO] In a real Sigstore flow:")
    print("         sigstore sign model_artifact.pt")
    print("       This produces a .sigstore bundle containing the Rekor log entry.")
    print("       Verification: sigstore verify model_artifact.pt")
    print("       No long-lived private keys to manage — the cert expires in ~10 minutes.\n")

    return bundle


def _compute_file_sha256(file_path: str) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# STEP 4 — Tamper detection
# ===========================================================================

class TamperDetectedError(Exception):  # noqa: F811
    """Raised when a model artifact's on-disk hash does not match the signing bundle."""
    pass


def verify_artifact(model_path: str, bundle_path: str) -> bool:
    """
    Verify that a model artifact has not been modified since it was signed.

    Algorithm:
      1. Read the expected hash from the signing bundle JSON.
      2. Recompute the SHA-256 of the model file on disk.
      3. Compare the two hashes using hmac.compare_digest() (constant-time).
      4. If they match: return True (artifact is intact).
      5. If they differ: raise TamperDetectedError (artifact is compromised).

    Parameters
    ----------
    model_path  : str — path to the model file to verify
    bundle_path : str — path to the signing bundle JSON

    Returns
    -------
    True if the artifact is unmodified.

    Raises
    ------
    TamperDetectedError  if hashes do not match
    FileNotFoundError    if either file is missing
    ValueError           if the bundle JSON is malformed
    """
    # Load the signing bundle
    bundle_text = Path(bundle_path).read_text()
    bundle      = json.loads(bundle_text)

    expected_hash = bundle.get("artifactHash", {}).get("digest")
    if not expected_hash:
        raise ValueError(f"Bundle at '{bundle_path}' is missing 'artifactHash.digest' field.")

    # Recompute the current hash of the model file
    actual_hash = _compute_file_sha256(model_path)

    # Constant-time comparison prevents timing-based side channels
    hashes_match = hmac.compare_digest(expected_hash, actual_hash)

    if not hashes_match:
        raise TamperDetectedError(
            f"\n  *** TAMPER DETECTED ***\n"
            f"  Model file  : {model_path}\n"
            f"  Bundle file : {bundle_path}\n"
            f"  Expected SHA-256: {expected_hash}\n"
            f"  Actual   SHA-256: {actual_hash}\n"
            f"  The file has been modified since it was signed. DO NOT USE IT."
        )

    return True


def step4_demonstrate_tamper_detection(
    model_path:  str = "model_artifact.pt",
    bundle_path: str = "model_artifact.pt.bundle.json",
) -> None:
    """
    Demonstrate tamper detection by:
      A) Verifying the original (unmodified) model — should PASS.
      B) Flipping one byte in the model file and verifying again — should FAIL.
      C) Restoring the original file.

    This shows that even a one-byte modification is caught by SHA-256 verification.
    """
    print("\n" + "#" * 60)
    print("# STEP 4 — Tamper Detection Demo")
    print("#" * 60)

    # --- Scenario A: Verify the original, unmodified model ---
    print("\n--- Scenario A: Verifying the ORIGINAL model_artifact.pt ---")
    try:
        result = verify_artifact(model_path, bundle_path)
        print("[PASS] Artifact verified successfully — no tampering detected.")
        print(f"       SHA-256 matches the signed bundle in {bundle_path}.")
    except TamperDetectedError as e:
        print(f"[FAIL] Unexpected tamper detection on clean file:\n{e}")
    except Exception as e:
        print(f"[ERROR] Verification failed unexpectedly: {e}")

    # --- Scenario B: Flip one byte and verify ---
    print("\n--- Scenario B: Flipping one byte in model_artifact.pt ---")

    # Read the original bytes so we can restore them after the demo
    original_bytes = Path(model_path).read_bytes()

    # Flip byte at position 512 (safely within the file for any model > 512 bytes)
    flip_position = min(512, len(original_bytes) - 1)
    tampered      = bytearray(original_bytes)
    original_byte = tampered[flip_position]
    tampered[flip_position] ^= 0xFF   # XOR with 0xFF flips all 8 bits

    print(f"[*] Flipping byte at position {flip_position}:")
    print(f"    Before: 0x{original_byte:02X}")
    print(f"    After : 0x{tampered[flip_position]:02X}")

    # Write the tampered version to disk
    Path(model_path).write_bytes(bytes(tampered))
    print(f"[*] Tampered file written to: {model_path}")

    # Run verification on the tampered file
    print(f"[*] Running verify_artifact('{model_path}', '{bundle_path}') ...")
    try:
        verify_artifact(model_path, bundle_path)
        print("[!] BUG: Tamper was not detected — verify_artifact should have raised an error!")
    except TamperDetectedError as e:
        print(f"\n[DETECTED] TamperDetectedError raised as expected:{e}")
        print(  "\n[+] Tamper detection is working correctly.")
        print(  "    Even a single-bit flip in the model file causes SHA-256 to produce")
        print(  "    a completely different digest (avalanche effect).")

    # --- Restore the original file ---
    print(f"\n[*] Restoring original model_artifact.pt ...")
    Path(model_path).write_bytes(original_bytes)
    print(f"[+] File restored. Final verification (should PASS):")

    try:
        verify_artifact(model_path, bundle_path)
        print("[PASS] Restored file verifies correctly.\n")
    except TamperDetectedError as e:
        print(f"[ERROR] Restore failed: {e}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Lab 7.2 — Dependency Auditing & Artifact Verification")
    print("  Data Poisoning Protection Course")
    print("=" * 60)

    # Determine paths relative to the script location for portability
    script_dir  = Path(__file__).parent
    req_input   = script_dir / "requirements.txt"
    req_output  = script_dir / "requirements_pinned.txt"
    model_path  = script_dir / "model_artifact.pt"
    bundle_path = script_dir / "model_artifact.pt.bundle.json"

    # Run all four steps
    vulns = step1_run_pip_audit()

    step2_generate_hash_pinned_requirements(
        input_path=str(req_input),
        output_path=str(req_output),
    )

    step3_sign_model_artifact(
        model_path=str(model_path),
        bundle_path=str(bundle_path),
    )

    step4_demonstrate_tamper_detection(
        model_path=str(model_path),
        bundle_path=str(bundle_path),
    )

    # --- Final summary ---
    print("=" * 60)
    print("  Lab 7.2 Complete")
    print("=" * 60)
    print("\nFiles created:")
    for fname in ["requirements_pinned.txt", "model_artifact.pt", "model_artifact.pt.bundle.json"]:
        p = script_dir / fname
        if p.exists():
            print(f"  {fname} ({p.stat().st_size} bytes)")

    print("\nKey takeaways:")
    print("  1. pip-audit surfaces known CVEs in your dependency tree automatically.")
    print("  2. Hash pinning closes the mirror-substitution gap that version pins leave open.")
    print("  3. Sigstore keyless signing eliminates long-lived signing keys entirely.")
    print("  4. SHA-256 + constant-time comparison catches any byte-level modification.")
    print("  5. All four controls together form a defence-in-depth supply-chain posture.\n")
