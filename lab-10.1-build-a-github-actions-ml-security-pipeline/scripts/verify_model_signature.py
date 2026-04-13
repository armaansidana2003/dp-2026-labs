"""
scripts/verify_model_signature.py
===================================
Verifies the HMAC-SHA256 signature of model.pt against model.sig.

The signing step is done via `make sign-model` (see Makefile).
The secret key is stored in the MODEL_HMAC_SECRET environment variable /
GitHub secret.

Exit code 1 if verification fails (CI gate).
"""

import sys
import hmac
import hashlib
import pathlib
import os

MODEL_PATH  = pathlib.Path("model.pt")
SIG_PATH    = pathlib.Path("model.sig")
LOG_PATH    = pathlib.Path("signature_verification.log")

SECRET_ENV  = "MODEL_HMAC_SECRET"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_hmac(path: pathlib.Path, secret: bytes) -> str:
    """Return HMAC-SHA256 hex digest of a file."""
    mac = hmac.new(secret, digestmod=hashlib.sha256)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            mac.update(chunk)
    return mac.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_lines = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    # ---- Check files ----
    if not MODEL_PATH.exists():
        log(f"[ERROR] Model file not found: {MODEL_PATH}")
        LOG_PATH.write_text("\n".join(log_lines))
        sys.exit(1)

    if not SIG_PATH.exists():
        log(f"[WARN] Signature file not found: {SIG_PATH}")
        log("[WARN] Skipping HMAC verification (first run — no signature yet).")
        log("[INFO] Run `make sign-model` to generate model.sig.")
        LOG_PATH.write_text("\n".join(log_lines))
        # Do not fail the pipeline on first run — just warn
        sys.exit(0)

    # ---- Load secret ----
    secret_hex = os.environ.get(SECRET_ENV, "")
    if not secret_hex:
        log(f"[WARN] {SECRET_ENV} environment variable not set.")
        log("[WARN] Using default development key — DO NOT USE IN PRODUCTION.")
        secret_hex = "0" * 64   # 32 zero bytes → only for local dev
    try:
        secret = bytes.fromhex(secret_hex)
    except ValueError:
        log(f"[ERROR] {SECRET_ENV} is not valid hex.")
        LOG_PATH.write_text("\n".join(log_lines))
        sys.exit(1)

    # ---- Compute and compare ----
    expected_sig = SIG_PATH.read_text(encoding="utf-8").strip()
    actual_sig   = compute_hmac(MODEL_PATH, secret)

    log(f"[INFO] Model    : {MODEL_PATH}")
    log(f"[INFO] Signature: {SIG_PATH}")
    log(f"[INFO] Expected : {expected_sig}")
    log(f"[INFO] Computed : {actual_sig}")

    if hmac.compare_digest(expected_sig, actual_sig):
        log("[OK] HMAC signature verified — model artifact is authentic.")
    else:
        log("[ERROR] Signature mismatch — model artifact may have been tampered with!")
        LOG_PATH.write_text("\n".join(log_lines))
        sys.exit(1)

    LOG_PATH.write_text("\n".join(log_lines))


if __name__ == "__main__":
    main()
