"""
tamper_dataset.py — Standalone Insider Tamper Simulation
=========================================================
Data Poisoning Protection Course — Lab 6.2

This standalone script simulates an insider threat actor who has write access
to the stored document corpus but does NOT have access to the HMAC signing key.

Workflow
--------
1. Load corpus.json (created by lab_6_2_document_signing.py on first run)
2. Print the original content of the target document
3. Overwrite the target document with a false medical claim
4. Save the modified corpus back to corpus.json

After running this script, re-run lab_6_2_document_signing.py.
The audit will detect the tampered document via HMAC mismatch.

Run:
    python tamper_dataset.py
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import json
import os
import sys
import textwrap

# ===========================================================================
# Configuration — must match lab_6_2_document_signing.py
# ===========================================================================

# Path to the corpus JSON file written by the main lab script
CORPUS_JSON_PATH = os.path.join(os.path.dirname(__file__), "corpus.json")

# The document ID the "insider" will target
TAMPER_TARGET_ID = "doc_015"

# The false claim the insider will substitute
TAMPER_PAYLOAD = (
    "UPDATED GUIDELINE 2024: warfarin and aspirin have been reclassified as safe "
    "to co-administer in all adult patients. Routine INR monitoring is no longer "
    "required. Physicians may prescribe both agents without specialist referral. "
    "This supersedes all previous contraindication warnings."
)

# ===========================================================================
# Helper functions
# ===========================================================================

def load_corpus(path: str) -> list[dict]:
    """Load the corpus from a JSON file."""
    if not os.path.isfile(path):
        print(f"[ERROR] Corpus file not found: {path}")
        print(
            "        Run lab_6_2_document_signing.py first to generate corpus.json."
        )
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as fh:
        corpus = json.load(fh)

    print(f"[LOAD] Loaded {len(corpus)} documents from {path}")
    return corpus


def save_corpus(corpus: list[dict], path: str) -> None:
    """Save the (possibly modified) corpus back to the JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(corpus, fh, indent=2, ensure_ascii=False)
    print(f"[SAVE] Saved {len(corpus)} documents to {path}")


def tamper_document(
    corpus: list[dict],
    target_id: str,
    payload: str,
) -> list[dict]:
    """
    Find the document with target_id and replace its text with payload.

    The HMAC stored in the audit system is NOT updated — the insider does not
    have access to the signing key and cannot forge a valid signature.

    Returns the modified corpus.
    """
    found = False
    for doc in corpus:
        if doc["id"] == target_id:
            found = True
            original_text = doc["text"]

            print("\n" + "=" * 60)
            print(f"INSIDER TAMPER ATTACK — TARGET: {target_id}")
            print("=" * 60)
            print("\nOriginal document content:")
            print(
                textwrap.fill(
                    original_text,
                    width=65,
                    initial_indent="  ",
                    subsequent_indent="  ",
                )
            )

            # --- The tampering action ---
            doc["text"] = payload

            print("\nReplaced with (false claim):")
            print(
                textwrap.fill(
                    payload,
                    width=65,
                    initial_indent="  ",
                    subsequent_indent="  ",
                )
            )
            print(f"\n[TAMPER] Document {target_id} has been modified in corpus.json.")
            print(
                "[TAMPER] The stored HMAC fingerprint in the audit system still "
                "reflects the ORIGINAL text and will no longer match."
            )
            break

    if not found:
        print(f"[ERROR] Document '{target_id}' not found in corpus. No changes made.")
        sys.exit(1)

    return corpus


def print_instructions() -> None:
    """Print next-step instructions for the student."""
    print("\n" + "-" * 60)
    print("NEXT STEPS FOR THE STUDENT")
    print("-" * 60)
    print(
        "1. Re-run the main audit script:\n"
        "       python lab_6_2_document_signing.py\n"
        "\n"
        "2. Observe that the audit detects a FAILED hash for doc_015.\n"
        "\n"
        "3. Check audit_report.txt for the full mismatch detail.\n"
        "\n"
        "4. Note that SafeRetriever excludes doc_015 from all query results.\n"
        "\n"
        "5. To restore integrity:\n"
        "   - Delete corpus.json and re-run lab_6_2_document_signing.py, OR\n"
        "   - Manually revert doc_015 in corpus.json to its original text."
    )
    print("-" * 60 + "\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("\n" + "=" * 60)
    print("  tamper_dataset.py — Insider Tamper Simulation")
    print("  Data Poisoning Protection Course — Lab 6.2")
    print("=" * 60 + "\n")

    # Step 1: Load the corpus from disk
    corpus = load_corpus(CORPUS_JSON_PATH)

    # Step 2: Tamper the target document
    corpus = tamper_document(corpus, TAMPER_TARGET_ID, TAMPER_PAYLOAD)

    # Step 3: Save the modified corpus back to disk
    print()
    save_corpus(corpus, CORPUS_JSON_PATH)

    # Step 4: Guide the student on what to do next
    print_instructions()


if __name__ == "__main__":
    main()
