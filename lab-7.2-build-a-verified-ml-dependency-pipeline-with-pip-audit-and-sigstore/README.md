# Lab 7.2 — Dependency Auditing & Artifact Signature Verification

## Overview

This lab covers the second pillar of ML supply-chain security: ensuring that every package your pipeline depends on is free of known CVEs, and that every model artifact you deploy is cryptographically verified to be unmodified. You will run automated dependency auditing, generate hash-pinned requirements, simulate Sigstore-style model signing, and build a tamper detection function that catches even a single flipped byte.

---

## Learning Objectives

By the end of this lab you will be able to:

1. Run `pip-audit` programmatically and interpret its JSON output.
2. Distinguish between pinned versions (`==`) and hash-pinned requirements (`--hash=sha256:`).
3. Explain why version pinning alone does not prevent dependency confusion or mirror-substitution attacks.
4. Describe how Sigstore keyless signing works at a high level (OIDC token, Fulcio CA, Rekor transparency log).
5. Implement SHA-256 based tamper detection for ML model artifacts.
6. Write a `verify_artifact()` function suitable for use in a deployment pre-flight check.

---

## Prerequisites

- Python 3.9 or later
- pip package manager
- Lab 7.1 completed (conceptual understanding of model supply-chain attacks)

Install dependencies before starting:

```
pip install -r requirements.txt
```

Note: `hashlib` is Python standard library and does not require installation. The `sigstore` package is included but this lab simulates signing with HMAC because real Sigstore requires an OIDC provider (GitHub Actions, Google, Microsoft). The simulation is clearly annotated.

---

## Lab Steps

### Step 1 — Automated Dependency Vulnerability Scanning with pip-audit

`pip-audit` queries the Python Packaging Advisory Database (PyPA) for known CVEs against every package in your current environment. The script runs it via subprocess with `--output json`, parses the result, and prints a formatted vulnerability table.

**What to observe:** Any packages with known CVEs appear in the table with their CVE ID and severity. If your environment is patched, the table will be empty — that is the desired outcome. The script still demonstrates the parsing logic.

### Step 2 — Hash-Pinned Requirements Generation

A standard `requirements.txt` with version pins (`torch==2.0.0`) prevents version drift but does not guarantee the exact wheel you download. A mirror could serve a different file with the same version string. Hash pinning ties the requirement to the exact file content — pip will reject any wheel whose SHA-256 does not match.

The script reads the current `requirements.txt`, generates SHA-256 hashes for each package entry (using `generate_pinned_requirements.py` logic), and writes `requirements_pinned.txt` in the pip hash-check format.

**What to observe:** The output file uses `--hash=sha256:...` suffixes. Running `pip install -r requirements_pinned.txt` with a tampered wheel will fail.

### Step 3 — Model Artifact Signing (Sigstore Simulation)

A real Sigstore keyless signing flow works as follows:
1. The signer authenticates to an OIDC provider (e.g., GitHub Actions OIDC).
2. Fulcio CA issues a short-lived signing certificate bound to the identity.
3. The artifact hash is signed with the certificate's private key.
4. The signature and certificate are submitted to the Rekor transparency log.

Because this lab runs locally without OIDC, we simulate the signing bundle using HMAC-SHA256. The simulation produces the same JSON bundle structure as a real Sigstore bundle, making the verification logic directly portable to a real implementation.

**What to observe:** A `model_artifact.pt.bundle.json` file is created containing the artifact hash, HMAC signature, timestamp, and signer identity. In a real deployment this file would be stored alongside the model or in the Rekor log.

### Step 4 — Tamper Detection

The `verify_artifact(model_path, bundle_path)` function recomputes the SHA-256 of the model file on disk and compares it to the hash stored in the signing bundle. If they differ by even one byte, a `TamperDetectedError` is raised.

The demo deliberately flips one byte in `model_artifact.pt`, then runs verification to show detection.

**What to observe:** The function catches the tampered file immediately and prints a clear warning. The original file is restored after the demo.

---

## Running the Lab

```bash
python lab_7_2_dependency_signing.py
```

For the standalone pinned-requirements generator:

```bash
python generate_pinned_requirements.py
```

This reads `requirements.txt` in the current directory and writes `requirements_pinned.txt`. Replace the generated hashes with real hashes obtained from `pip download` before using in production.

---

## Expected Outputs

| Output File | Contents |
|---|---|
| `requirements_pinned.txt` | Hash-pinned pip requirements in `--hash=sha256:` format |
| `model_artifact.pt` | Simple PyTorch MLP weights |
| `model_artifact.pt.bundle.json` | Signing bundle with artifact hash, HMAC signature, timestamp |

Console output includes:
- pip-audit vulnerability table (empty if environment is patched)
- Hash-pinning generation confirmation
- Signing bundle creation confirmation
- Tamper detection trigger showing `TamperDetectedError`

---

## Discussion Questions

1. pip-audit flags vulnerabilities in currently installed packages. Why is it important to also run it as part of the CI/CD pipeline during dependency installation, not just as a one-off check?

2. Hash pinning ties you to a specific wheel file. What operational challenges does this create when you need to update a dependency — and what tooling helps manage this?

3. Real Sigstore uses short-lived certificates and a transparency log (Rekor). Why are short-lived certificates preferable to long-lived signing keys for artifact signing in a CI/CD context?

4. The tamper detection in Step 4 catches even a one-byte change. A sophisticated attacker might try to find a hash collision instead. What does this tell you about SHA-256 and the practical limits of hash-based integrity?

5. The HMAC signing in this lab uses a hardcoded secret key. What is the correct way to manage this secret in a production environment, and what additional controls should surround it?

6. If a compromised dependency (malicious package) was installed before your hash-pinning was in place, would hash-pinning detect it retroactively? Why or why not?
