"""
scripts/check_data_integrity.py
================================
Verifies SHA-256 hashes of data files against data/manifest.json.
Exits with code 1 if any file hash does not match (CI gate).
"""

import sys
import json
import hashlib
import pathlib

MANIFEST_PATH = pathlib.Path("data/manifest.json")
DATA_DIR      = pathlib.Path("data")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of_file(path: pathlib.Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not MANIFEST_PATH.exists():
        print(f"[WARN] {MANIFEST_PATH} not found — generating manifest from current files.")
        # Auto-generate manifest if running for the first time
        manifest = {}
        for csv_file in DATA_DIR.glob("*.csv"):
            digest = sha256_of_file(csv_file)
            manifest[str(csv_file)] = digest
            print(f"  [MANIFEST] {csv_file}: {digest}")
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
        print(f"[INFO] Manifest written to {MANIFEST_PATH}")
        print("[OK] Integrity check skipped on first run (manifest just created).")
        sys.exit(0)

    manifest = json.loads(MANIFEST_PATH.read_text())
    print(f"[INFO] Loaded manifest with {len(manifest)} file entries.")

    all_ok = True

    for file_path_str, expected_hash in manifest.items():
        file_path = pathlib.Path(file_path_str)

        if not file_path.exists():
            print(f"  [FAIL] File missing: {file_path}")
            all_ok = False
            continue

        actual_hash = sha256_of_file(file_path)

        if actual_hash == expected_hash:
            print(f"  [PASS] {file_path}  SHA-256 match.")
        else:
            print(f"  [FAIL] {file_path}")
            print(f"         Expected : {expected_hash}")
            print(f"         Got      : {actual_hash}")
            all_ok = False

    if not all_ok:
        print("\n[ERROR] Data integrity check FAILED — possible tampering detected.")
        sys.exit(1)

    print(f"\n[OK] All {len(manifest)} data files passed integrity check.")


if __name__ == "__main__":
    main()
