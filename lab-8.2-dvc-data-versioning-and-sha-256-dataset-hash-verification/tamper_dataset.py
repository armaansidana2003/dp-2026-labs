"""
tamper_dataset.py — Standalone Tamper Simulation
=================================================
Course : Data Poisoning Protection — Lab 8.2
Author : Armaan Sidana

Simulates an adversary who obtains write access to the dataset file
and flips 50 income labels to corrupt the training signal.

Usage:
    python tamper_dataset.py

After running this script, execute:
    python lab_8_2_dvc_versioning.py   # to detect and recover
or manually:
    python -c "
    import lab_8_2_dvc_versioning as lab, pathlib
    lab.dvc_status(pathlib.Path('adult_train.csv'))
    "
"""

import sys
import pathlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR  = pathlib.Path(__file__).parent.resolve()
DATA_FILE   = SCRIPT_DIR / "adult_train.csv"
N_TAMPER    = 50
RANDOM_SEED = 777   # Different seed from the main lab so the flip set differs


def tamper(filepath: pathlib.Path, n_tamper: int, seed: int) -> None:
    """
    Flip `n_tamper` randomly selected income labels in `filepath`.

    The attack is:
    - Low-visibility: only 5 % of 1 000 rows are changed.
    - Targeted: flips 0→1 (promote low earner to high earner) to
      inject noise into the positive class boundary.
    """
    if not filepath.exists():
        print(f"[ERROR] Dataset not found: {filepath}")
        print("        Run lab_8_2_dvc_versioning.py first to generate it.")
        sys.exit(1)

    # Load dataset
    df = pd.read_csv(filepath)
    print(f"[*] Loaded {filepath.name}: {len(df)} rows")
    print(f"    Income before: {df['income'].value_counts().sort_index().to_dict()}")

    # Choose rows to flip
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n_tamper, len(df)), replace=False)

    # Apply label flip (0→1, 1→0)
    df.loc[idx, "income"] = 1 - df.loc[idx, "income"]

    # Save back to same path (in-place attack)
    df.to_csv(filepath, index=False)
    print(f"    Income after : {df['income'].value_counts().sort_index().to_dict()}")
    print(f"\n[!] Tampered {len(idx)} records in {filepath.name}.")
    print("    Run dvc_status to detect the modification:")
    print()
    print("    python -c \"")
    print("    import lab_8_2_dvc_versioning as lab, pathlib")
    print("    lab.dvc_status(pathlib.Path('adult_train.csv'))\"")


if __name__ == "__main__":
    print("=" * 60)
    print(" Standalone Tamper Simulation — Lab 8.2")
    print("=" * 60)
    print()
    tamper(DATA_FILE, N_TAMPER, RANDOM_SEED)
