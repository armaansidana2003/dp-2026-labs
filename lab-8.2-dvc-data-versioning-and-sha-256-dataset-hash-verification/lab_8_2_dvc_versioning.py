"""
Lab 8.2 — DVC Dataset Versioning & Integrity
=============================================
Course : Data Poisoning Protection
Author : Armaan Sidana

This script simulates a DVC (Data Version Control) workflow entirely
in pure Python — no Git repository or DVC remote is required.

Workflow
--------
1. Generate adult_train.csv
2. dvc_add()  → compute SHA-256, write .dvc metadata, write manifest
3. Back up original to .dvc_cache/
4. tamper_dataset() → flip 50 income labels (simulate poisoning)
5. dvc_status() → detect MODIFIED (hash mismatch)
6. dvc_checkout() → restore from cache
7. dvc_status() → confirm OK

Run:
    python lab_8_2_dvc_versioning.py
"""

import sys
import os
import json
import shutil
import hashlib
import pathlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (all relative to the script's directory so the lab is self-contained)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = pathlib.Path(__file__).parent.resolve()
DATA_FILE    = SCRIPT_DIR / "adult_train.csv"
DVC_FILE     = SCRIPT_DIR / "adult_train.csv.dvc"
MANIFEST_FILE = SCRIPT_DIR / "manifest.json"
CACHE_DIR    = SCRIPT_DIR / ".dvc_cache"

RANDOM_SEED  = 42
N_ROWS       = 1000
N_TAMPER     = 50


# ===========================================================================
# SECTION 1 — Dataset Generation
# ===========================================================================

def generate_dataset(filepath: pathlib.Path, n: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic version of the UCI Adult dataset.

    Columns
    -------
    age           : int,  18 – 90
    hours_per_week: int,  1 – 99
    education_num : int,  1 – 16
    income        : int,  0 = <=50K, 1 = >50K
    """
    rng = np.random.default_rng(seed)

    age            = rng.integers(18, 91, size=n)
    hours_per_week = rng.integers(1, 100, size=n)
    education_num  = rng.integers(1, 17, size=n)

    # Income loosely correlated with education
    prob_high = (education_num - 1) / 15 * 0.6 + 0.1
    income    = (rng.random(size=n) < prob_high).astype(int)

    df = pd.DataFrame({
        "age":            age.astype(int),
        "hours_per_week": hours_per_week.astype(int),
        "education_num":  education_num.astype(int),
        "income":         income,
    })

    df.to_csv(filepath, index=False)
    return df


# ===========================================================================
# SECTION 2 — Hash Utilities
# ===========================================================================

def compute_sha256(filepath: pathlib.Path) -> str:
    """
    Compute the SHA-256 hash of a file by reading it in chunks.
    This approach handles large files without loading them into memory.

    Returns the hex digest string (64 characters).
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as fh:
        # Read in 64 KB chunks
        for chunk in iter(lambda: fh.read(65_536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ===========================================================================
# SECTION 3 — DVC-like Operations
# ===========================================================================

def dvc_add(filepath: pathlib.Path) -> dict:
    """
    Simulate `dvc add <filepath>`.

    Actions
    -------
    1. Compute SHA-256 of the file.
    2. Write a YAML-format .dvc metadata file (mirrors DVC's real format).
    3. Write / update manifest.json (our lab's version of the DVC lockfile).

    Returns the metadata dict.
    """
    filepath = pathlib.Path(filepath)
    size     = filepath.stat().st_size
    digest   = compute_sha256(filepath)

    # --- Write .dvc file (real DVC uses YAML) ---
    dvc_path = filepath.parent / (filepath.name + ".dvc")
    dvc_content = (
        "# DVC metadata file — do not edit manually\n"
        "outs:\n"
        f"- md5: {digest}\n"
        f"  size: {size}\n"
        f"  path: {filepath.name}\n"
    )
    dvc_path.write_text(dvc_content)

    # --- Write / update manifest.json ---
    manifest_path = filepath.parent / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as mf:
            manifest = json.load(mf)

    manifest[filepath.name] = {
        "sha256": digest,
        "size":   size,
        "path":   str(filepath),
    }
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    return {"path": str(filepath), "sha256": digest, "size": size}


def dvc_status(filepath: pathlib.Path) -> str:
    """
    Simulate `dvc status` for a single tracked file.

    Returns "OK" if the current hash matches the stored hash,
    or "MODIFIED" if they differ.

    Prints detailed hash comparison either way.
    """
    filepath = pathlib.Path(filepath)
    dvc_path = filepath.parent / (filepath.name + ".dvc")

    if not dvc_path.exists():
        print(f"  [ERROR] No .dvc file found for {filepath.name}. Run dvc_add first.")
        return "UNTRACKED"

    # Parse the stored hash from the .dvc YAML
    stored_hash = None
    for line in dvc_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("md5:"):
            stored_hash = stripped.split(":", 1)[1].strip()
            break

    if stored_hash is None:
        print("  [ERROR] Could not parse md5 from .dvc file.")
        return "ERROR"

    current_hash = compute_sha256(filepath)

    if current_hash == stored_hash:
        status = "OK"
        symbol = "\u2713"   # ✓
    else:
        status = "MODIFIED"
        symbol = "\u2717"   # ✗

    print(f"  [{symbol}] {filepath.name}  {status}")
    print(f"        stored : {stored_hash[:16]}...{stored_hash[-8:]}")
    print(f"        current: {current_hash[:16]}...{current_hash[-8:]}")

    return status


def dvc_checkout(filepath: pathlib.Path, cache_dir: pathlib.Path) -> bool:
    """
    Simulate `dvc checkout` — restore a file from the local DVC cache.

    The cache stores files by their SHA-256 hash so that multiple
    versions of the same logical file can coexist.

    Returns True if restoration succeeded, False otherwise.
    """
    filepath  = pathlib.Path(filepath)
    cache_dir = pathlib.Path(cache_dir)
    dvc_path  = filepath.parent / (filepath.name + ".dvc")

    if not dvc_path.exists():
        print(f"  [ERROR] No .dvc file found for {filepath.name}.")
        return False

    # Read stored hash from .dvc file
    stored_hash = None
    for line in dvc_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("md5:"):
            stored_hash = stripped.split(":", 1)[1].strip()
            break

    cached_path = cache_dir / stored_hash
    if not cached_path.exists():
        print(f"  [ERROR] Cache entry not found: {cached_path}")
        return False

    # Restore file
    shutil.copy2(cached_path, filepath)
    print(f"  [+] Restored {filepath.name} from .dvc_cache/{stored_hash[:16]}...")
    return True


def backup_to_cache(filepath: pathlib.Path, cache_dir: pathlib.Path) -> pathlib.Path:
    """
    Copy the file to the local DVC cache, named by its SHA-256 hash.
    This is what `dvc add` does internally when caching a new file.
    """
    filepath  = pathlib.Path(filepath)
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    digest      = compute_sha256(filepath)
    cached_path = cache_dir / digest
    shutil.copy2(filepath, cached_path)
    return cached_path


# ===========================================================================
# SECTION 4 — Tamper Simulation
# ===========================================================================

def tamper_dataset(filepath: pathlib.Path, n_tamper: int = N_TAMPER, seed: int = 99) -> int:
    """
    Simulate a label-flip poisoning attack:
    randomly flip `n_tamper` income labels in the CSV file.

    Returns the number of rows actually flipped.
    """
    filepath = pathlib.Path(filepath)
    df  = pd.read_csv(filepath)
    rng = np.random.default_rng(seed)

    idx = rng.choice(len(df), size=min(n_tamper, len(df)), replace=False)
    df.loc[idx, "income"] = 1 - df.loc[idx, "income"]
    df.to_csv(filepath, index=False)
    return len(idx)


# ===========================================================================
# SECTION 5 — Main Demo
# ===========================================================================

def main():
    # Track pass/fail for final summary
    results: dict[str, bool] = {}

    print("=" * 65)
    print(" Lab 8.2 — DVC Dataset Versioning & Integrity")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 1: Generate dataset
    # ------------------------------------------------------------------
    print("\n--- Step 1: Generate Dataset ---")
    df = generate_dataset(DATA_FILE)
    print(f"[+] Generated {DATA_FILE.name} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"    Columns    : {list(df.columns)}")
    print(f"    Income dist: {df['income'].value_counts().to_dict()}")
    results["Generate dataset"] = DATA_FILE.exists()

    # ------------------------------------------------------------------
    # Step 2: DVC Add — record hash baseline
    # ------------------------------------------------------------------
    print("\n--- Step 2: DVC Add (record hash baseline) ---")
    meta = dvc_add(DATA_FILE)
    print(f"[+] SHA-256  : {meta['sha256'][:32]}...")
    print(f"[+] File size: {meta['size']:,} bytes")
    print(f"[+] Written  : {DVC_FILE.name}")
    print(f"[+] Written  : {MANIFEST_FILE.name}")
    results["DVC add (hash)"] = DVC_FILE.exists() and MANIFEST_FILE.exists()

    # ------------------------------------------------------------------
    # Step 3: Backup original to DVC cache
    # ------------------------------------------------------------------
    print("\n--- Step 3: Backup to DVC Cache ---")
    cached = backup_to_cache(DATA_FILE, CACHE_DIR)
    print(f"[+] Cached → {CACHE_DIR.name}/{cached.name[:32]}...")
    results["Backup to cache"] = cached.exists()

    # ------------------------------------------------------------------
    # Step 4: Tamper the dataset
    # ------------------------------------------------------------------
    print("\n--- Step 4: Tamper Dataset ---")
    n_flipped = tamper_dataset(DATA_FILE)
    print(f"[\u0021] Flipped {n_flipped} income labels in {DATA_FILE.name}")
    print(f"    New SHA-256: {compute_sha256(DATA_FILE)[:32]}...")

    # ------------------------------------------------------------------
    # Step 5: DVC Status — detect tampering
    # ------------------------------------------------------------------
    print("\n--- Step 5: DVC Status (detect tampering) ---")
    status_before = dvc_status(DATA_FILE)
    detected = (status_before == "MODIFIED")
    results["Tamper detection"] = detected
    if detected:
        print("    \u2192 Poisoning attack detected! Blocking pipeline.")
    else:
        print("    \u2192 [WARNING] Tamper NOT detected — review hashing logic.")

    # ------------------------------------------------------------------
    # Step 6: DVC Checkout — restore from cache
    # ------------------------------------------------------------------
    print("\n--- Step 6: DVC Checkout (restore clean file) ---")
    restored = dvc_checkout(DATA_FILE, CACHE_DIR)
    results["Checkout/restore"] = restored
    if restored:
        print(f"    Restored income dist: {pd.read_csv(DATA_FILE)['income'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # Step 7: Confirm status is OK after restoration
    # ------------------------------------------------------------------
    print("\n--- Step 7: DVC Status (confirm restoration) ---")
    status_after = dvc_status(DATA_FILE)
    clean_again  = (status_after == "OK")
    results["Post-restore check"] = clean_again

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" SUMMARY")
    print("=" * 65)
    all_passed = True
    for step, passed in results.items():
        symbol = "\u2713" if passed else "\u2717"
        print(f"  {step:<26}: {symbol}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("[OK] All steps passed.")
        print("     In production, run dvc_status() as a pre-training CI check.")
    else:
        print("[FAIL] One or more steps failed. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
