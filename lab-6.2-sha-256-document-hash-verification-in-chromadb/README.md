# Lab 6.2 — Document Signing & Integrity Verification

## Overview
In this lab you will build a complete corpus integrity pipeline: sign a medical
document corpus with HMAC-SHA256, simulate an insider tampering attack, audit
the corpus to detect the tampered document, and generate a structured audit
report.

---

## Learning Objectives
1. Implement HMAC-SHA256 document signing for a vector-store corpus.
2. Store and retrieve cryptographic fingerprints from ChromaDB metadata.
3. Simulate an insider threat that modifies a stored document.
4. Run an automated audit that detects the mismatch and excludes the tampered
   document from retrieval results.
5. Generate a machine-readable audit report for compliance workflows.

---

## Prerequisites
- Python 3.10+
- A virtual environment (recommended)

```
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step-by-Step Instructions

### Step 1 — Run the main lab script
```
python lab_6_2_document_signing.py
```
This script:
- Builds a ChromaDB collection from 50 synthetic medical documents.
- Signs every document with HMAC-SHA256 and stores fingerprints both in a
  local dict and in ChromaDB metadata.
- Simulates an insider attack that replaces document `doc_015` with a false
  claim.
- Audits the full corpus and detects the tamper.
- Prints a detailed report to the console and saves `audit_report.txt`.

### Step 2 — Inspect the audit report
```
cat audit_report.txt
```
Review which documents passed, which failed, and the specific hash mismatch
detail for `doc_015`.

### Step 3 — Run the standalone tamper simulation
```
python tamper_dataset.py
```
This writes a modified `corpus.json`. Then re-run the main script to see
the audit catch the new tamper.

### Step 4 — Restore integrity and re-audit
Edit `corpus.json` to revert `doc_015` to its original content (or delete
`corpus.json` and let the main script regenerate it), then re-run:
```
python lab_6_2_document_signing.py
```
Confirm that all documents now pass verification.

---

## Expected Output (abridged)
```
[SIGN] Signing 50 documents...
[SIGN] All documents signed and ingested.

[TAMPER] Modifying doc_015 to insert false claim...
[TAMPER] doc_015 content replaced.

[AUDIT] Verifying corpus integrity...
[AUDIT] doc_000 ... OK
...
[AUDIT] doc_015 ... FAILED (hash mismatch)
...
[AUDIT] doc_049 ... OK

============================================================
CORPUS INTEGRITY REPORT
============================================================
Total documents  : 50
Verified OK      : 49
Failed / Tampered: 1
Tampered IDs     : ['doc_015']
============================================================
[REPORT] Audit report saved to audit_report.txt
```

---

## Discussion Questions
1. The HMAC key is hard-coded as `b"course-secret-key"`. In a production
   system, how would you manage this key securely?
2. An insider with write access to the database could also update the stored
   hash to match the tampered document. How would you prevent this?
3. What is the difference between HMAC-SHA256 and a digital signature
   (e.g. RSA/ECDSA)? When would you prefer one over the other for corpus
   integrity?
4. The audit report is saved as plain text. What format and additional fields
   would make it suitable for a SOC or compliance team?
5. If documents are updated legitimately (e.g. drug guidelines change), describe
   the re-signing workflow that maintains integrity without treating every update
   as a tamper event.
