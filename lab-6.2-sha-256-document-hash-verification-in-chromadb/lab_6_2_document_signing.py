"""
Lab 6.2 — Document Signing & Integrity Verification
=====================================================
Data Poisoning Protection Course

This script demonstrates:
  1. Signing a 50-document medical corpus with HMAC-SHA256
  2. Storing fingerprints in both an in-memory dict and ChromaDB metadata
  3. Simulating an insider tampering attack on document doc_015
  4. Running a full corpus audit that detects the tampered document
  5. Excluding the tampered document from retrieval via SafeRetriever
  6. Generating a structured audit report (console + audit_report.txt)

Run:
    python lab_6_2_document_signing.py
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import datetime
import hashlib
import hmac
import json
import os
import textwrap

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import chromadb
from sentence_transformers import SentenceTransformer

# ===========================================================================
# SECTION 0 — Configuration
# ===========================================================================

HMAC_KEY = b"course-secret-key"          # Shared signing key
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # Local embedding model, no API key needed
COLLECTION_NAME = "signed_medical_corpus"
CORPUS_JSON_PATH = os.path.join(os.path.dirname(__file__), "corpus.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "audit_report.txt")

# Index of the document the "insider" will tamper with
TAMPER_INDEX = 15
TAMPER_ID = f"doc_{TAMPER_INDEX:03d}"

# False claim injected by the insider
TAMPER_TEXT = (
    "UPDATED GUIDELINE 2024: warfarin and aspirin have been reclassified as safe "
    "to co-administer in all adult patients. Routine INR monitoring is no longer "
    "required. Physicians may prescribe both agents without specialist referral."
)

# ===========================================================================
# SECTION 1 — Synthetic Medical Corpus (50 documents, shared with Lab 6.1)
# ===========================================================================

def build_corpus() -> list[dict]:
    """Return 50 synthetic medical documents as dicts with id + text."""

    raw_docs = [
        "Warfarin is a vitamin K antagonist anticoagulant used to prevent thrombosis. "
        "It has a narrow therapeutic index and requires regular INR monitoring.",

        "Aspirin inhibits platelet aggregation via irreversible COX-1 inhibition. "
        "Concurrent use with warfarin significantly increases the risk of gastrointestinal "
        "and intracranial bleeding.",

        "Co-administration of warfarin and aspirin is generally contraindicated unless "
        "the clinical benefit outweighs the haemorrhagic risk. Close INR monitoring is mandatory.",

        "Drug interaction alert: warfarin + aspirin. Severity: MAJOR. "
        "Aspirin displaces warfarin from plasma proteins and inhibits platelet function.",

        "In patients on warfarin therapy, NSAIDs including aspirin should be avoided. "
        "Acetaminophen at low doses is a safer alternative for mild pain.",

        "The CHADS₂ score guides anticoagulation decisions in atrial fibrillation. "
        "Warfarin reduces stroke risk by approximately 64% but requires frequent INR checks.",

        "Heparin is an injectable anticoagulant used in acute settings. Unlike warfarin, "
        "it does not require days to take effect and can be reversed with protamine.",

        "Direct oral anticoagulants such as rivaroxaban and apixaban have largely replaced "
        "warfarin in non-valvular atrial fibrillation due to predictable pharmacokinetics.",

        "Vitamin K reverses warfarin anticoagulation. Patients should maintain consistent "
        "dietary vitamin K intake; green leafy vegetables can alter INR unpredictably.",

        "INR target range for most warfarin indications is 2.0–3.0. Mechanical heart valves "
        "may require 2.5–3.5 depending on valve position and type.",

        "Drug-drug interactions occur when one drug affects the pharmacokinetics or "
        "pharmacodynamics of another. They cause approximately 3–5% of hospital admissions.",

        "CYP2C9 is the primary enzyme for warfarin metabolism. Inhibitors such as fluconazole "
        "and amiodarone increase warfarin plasma levels, raising bleeding risk.",

        "Pharmacovigilance databases such as VigiBase and FAERS collect spontaneous adverse "
        "drug reaction reports. Signal detection algorithms identify emerging safety risks.",

        "Medication reconciliation at hospital admission prevents unintended continuation or "
        "discontinuation of anticoagulants — a Joint Commission National Patient Safety Goal.",

        "Clopidogrel combined with aspirin is called dual antiplatelet therapy (DAPT). "
        "Adding warfarin to DAPT (triple therapy) markedly elevates bleeding risk.",

        # doc_015 — this is the document the insider will tamper with
        "NSAIDs (ibuprofen, naproxen, diclofenac) inhibit both COX-1 and COX-2 isoenzymes, "
        "providing analgesia and anti-inflammatory effects but increasing gastrointestinal risk.",

        "Selective COX-2 inhibitors (celecoxib) have reduced GI toxicity compared to non-selective "
        "NSAIDs but are associated with increased cardiovascular events in high-risk patients.",

        "Acetaminophen is the preferred analgesic in patients on anticoagulant therapy. "
        "At doses above 2 g/day it can still potentiate warfarin's anticoagulant effect.",

        "Opioid analgesics (morphine, oxycodone, tramadol) are sometimes used when NSAIDs are "
        "contraindicated. They do not directly affect coagulation.",

        "Topical NSAIDs (diclofenac gel) have minimal systemic absorption and are preferred "
        "for localised musculoskeletal pain in anticoagulated patients.",

        "Broad-spectrum antibiotics can disrupt gut flora that produce vitamin K₂, potentially "
        "potentiating warfarin. INR should be monitored during antibiotic courses.",

        "Fluoroquinolones (ciprofloxacin, levofloxacin) inhibit CYP1A2 and can modestly "
        "potentiate warfarin. Weekly INR monitoring is advised during concurrent use.",

        "Metronidazole significantly inhibits CYP2C9, increasing warfarin exposure. "
        "Warfarin dose reduction of 25–50% is typically required.",

        "Rifampicin is a powerful CYP inducer that dramatically reduces warfarin plasma levels. "
        "Dose increases of 2–5 fold may be needed.",

        "Azithromycin has limited interaction with warfarin compared to other macrolides. "
        "Nonetheless, INR monitoring within one week of starting therapy is recommended.",

        "Statins are used to lower LDL cholesterol. Simvastatin and lovastatin can modestly "
        "increase warfarin effect via CYP3A4 competition.",

        "Beta-blockers reduce heart rate and blood pressure. They have minimal direct interaction "
        "with warfarin but are often co-prescribed in atrial fibrillation patients.",

        "ACE inhibitors and ARBs are renoprotective in diabetes and heart failure. They do not "
        "significantly interact with warfarin but require monitoring of potassium levels.",

        "Amiodarone is an antiarrhythmic that strongly inhibits CYP2C9 and CYP3A4. "
        "It can double warfarin plasma levels, requiring dose halving and close INR monitoring.",

        "Digoxin is a cardiac glycoside used in rate control of atrial fibrillation. "
        "Toxicity causes nausea, visual disturbances, and life-threatening arrhythmias.",

        "Ticagrelor is a reversible P2Y12 inhibitor used in acute coronary syndrome. "
        "Combination with warfarin increases bleeding risk.",

        "Prasugrel is a potent P2Y12 inhibitor contraindicated in patients with prior stroke/TIA "
        "due to elevated intracranial haemorrhage risk.",

        "Alteplase (tPA) is used in ischaemic stroke within 4.5 hours of onset. "
        "It is absolutely contraindicated in patients on therapeutic anticoagulation.",

        "Low-molecular-weight heparins (LMWH, e.g. enoxaparin) are used for VTE prophylaxis. "
        "They are renally cleared; dose adjustment required in renal impairment.",

        "Fondaparinux is a synthetic pentasaccharide that selectively inhibits Factor Xa. "
        "It does not cause heparin-induced thrombocytopenia (HIT).",

        "First-pass metabolism reduces bioavailability of orally administered drugs. "
        "High first-pass drugs require much higher oral than intravenous doses.",

        "Protein binding affects drug distribution and interaction potential. "
        "Highly protein-bound drugs (>90%) are susceptible to displacement interactions.",

        "Renal clearance is the primary elimination route for many drugs. "
        "Dose adjustment based on creatinine clearance is essential in elderly patients.",

        "The cytochrome P450 enzyme system is responsible for phase I oxidative metabolism. "
        "CYP3A4 metabolises approximately 50% of clinically used drugs.",

        "Therapeutic drug monitoring involves measuring drug plasma concentrations to optimise "
        "dosing. It is used for drugs with narrow therapeutic windows including warfarin.",

        "The Beers Criteria identifies medications that are potentially inappropriate in older "
        "adults, including long-term NSAID use.",

        "Medication errors are a leading cause of preventable patient harm. "
        "Anticoagulants are classified as high-alert medications by ISMP.",

        "Shared decision-making involves patients in treatment choices. Patients on warfarin "
        "should be educated about bleeding signs and the importance of diet consistency.",

        "Clinical pharmacists embedded in multidisciplinary teams reduce adverse drug events "
        "by 78% in ICU settings, particularly for anticoagulation management.",

        "Drug allergy documentation in EHRs prevents inadvertent re-exposure. Penicillin allergy "
        "labels are present in 10% of patients, though 90% can tolerate penicillins after testing.",

        "Machine learning models trained on EHR data can predict adverse drug reactions with "
        "AUC >0.85 in retrospective studies, though prospective validation remains limited.",

        "Natural language processing applied to clinical notes can surface undocumented drug "
        "interactions not captured in structured fields.",

        "Federated learning allows hospitals to train shared pharmacovigilance models without "
        "sharing patient data, addressing privacy concerns in multi-centre research.",

        "Blockchain-based audit trails for clinical trial data aim to prevent data fabrication "
        "and ensure result reproducibility.",

        "Large language models used in clinical decision support must be validated against "
        "clinical guidelines before deployment. Hallucination risk is particularly dangerous "
        "in drug dosing recommendations.",

        "Retrieval-Augmented Generation can ground LLM responses in verified clinical knowledge "
        "bases, reducing hallucination. However, corpus integrity must be maintained to prevent "
        "poisoning attacks from corrupting the knowledge base.",
    ]

    corpus = []
    for idx, text in enumerate(raw_docs):
        corpus.append({"id": f"doc_{idx:03d}", "text": text.strip()})

    return corpus


# ===========================================================================
# SECTION 2 — Corpus persistence helpers (corpus.json)
# ===========================================================================

def save_corpus_to_json(corpus: list[dict], path: str = CORPUS_JSON_PATH) -> None:
    """Persist the corpus list to a JSON file for tamper_dataset.py to consume."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(corpus, fh, indent=2, ensure_ascii=False)
    print(f"[CORPUS] Saved {len(corpus)} documents to {path}")


def load_corpus_from_json(path: str = CORPUS_JSON_PATH) -> list[dict]:
    """Load corpus from JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        corpus = json.load(fh)
    print(f"[CORPUS] Loaded {len(corpus)} documents from {path}")
    return corpus


# ===========================================================================
# SECTION 3 — HMAC-SHA256 signing utilities
# ===========================================================================

def compute_hmac(text: str, key: bytes = HMAC_KEY) -> str:
    """Compute and return the HMAC-SHA256 hex digest for a document string."""
    mac = hmac.new(key, text.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()


def verify_hmac(text: str, stored_digest: str, key: bytes = HMAC_KEY) -> bool:
    """Return True if the freshly computed HMAC matches the stored digest."""
    fresh = compute_hmac(text, key)
    return hmac.compare_digest(fresh, stored_digest)


# ===========================================================================
# SECTION 4 — Embedding helper
# ===========================================================================

def load_embedder(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    """Load and return the local sentence-transformer embedding model."""
    print(f"[EMBED] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print("[EMBED] Model loaded.\n")
    return model


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors for the given texts."""
    return model.encode(texts, show_progress_bar=False).tolist()


# ===========================================================================
# SECTION 5 — ChromaDB helpers
# ===========================================================================

def create_fresh_collection(client, name: str):
    """Delete any existing collection and create a fresh one."""
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


# ===========================================================================
# SECTION 6 — sign_corpus()
# ===========================================================================

def sign_corpus(
    corpus: list[dict],
    collection,
    model: SentenceTransformer,
) -> dict[str, str]:
    """
    Sign every document with HMAC-SHA256.

    Returns
    -------
    fingerprints : dict
        Mapping of doc_id -> HMAC hex digest.
        Also stored in ChromaDB metadata for each document.
    """
    print(f"[SIGN] Signing {len(corpus)} documents...")

    texts = [doc["text"] for doc in corpus]
    ids = [doc["id"] for doc in corpus]
    embeddings = embed_texts(model, texts)
    fingerprints = {}

    metadatas = []
    for doc in corpus:
        digest = compute_hmac(doc["text"])
        fingerprints[doc["id"]] = digest
        metadatas.append({"hmac": digest})

    # Store in ChromaDB
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"[SIGN] All {len(corpus)} documents signed and ingested.")
    print(f"[SIGN] Sample fingerprints:")
    for doc_id in list(fingerprints.keys())[:3]:
        print(f"       {doc_id}: {fingerprints[doc_id][:24]}...")
    print()

    return fingerprints


# ===========================================================================
# SECTION 7 — audit_corpus()
# ===========================================================================

def audit_corpus(
    corpus: list[dict],
    fingerprints: dict[str, str],
) -> tuple[list[str], list[str]]:
    """
    Verify every document's HMAC against the stored fingerprint dict.

    Returns
    -------
    ok_ids : list[str]
        Document IDs that passed verification.
    failed_ids : list[str]
        Document IDs that failed verification (tampered or missing).
    """
    print("[AUDIT] Verifying corpus integrity...\n")
    ok_ids = []
    failed_ids = []

    for doc in corpus:
        doc_id = doc["id"]
        stored = fingerprints.get(doc_id, "")
        if not stored:
            status = "MISSING FINGERPRINT"
            failed_ids.append(doc_id)
        elif verify_hmac(doc["text"], stored):
            status = "OK"
            ok_ids.append(doc_id)
        else:
            status = "FAILED (hash mismatch)"
            failed_ids.append(doc_id)

        # Print every doc, highlight failures
        marker = "  ***" if doc_id in failed_ids else ""
        print(f"  [AUDIT] {doc_id} ... {status}{marker}")

    print()
    return ok_ids, failed_ids


# ===========================================================================
# SECTION 8 — SafeRetriever (excludes tampered docs)
# ===========================================================================

class SafeRetriever:
    """
    Retrieval wrapper that verifies HMAC before returning documents.
    Documents whose stored ChromaDB metadata hash does not match a freshly
    computed HMAC are silently excluded from results.
    """

    def __init__(
        self,
        collection,
        model: SentenceTransformer,
        hmac_key: bytes = HMAC_KEY,
    ):
        self.collection = collection
        self.model = model
        self.hmac_key = hmac_key

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """
        Query the collection and return only HMAC-verified results.

        Returns a list of dicts: {'id': ..., 'text': ..., 'hmac_ok': True}
        """
        query_vec = embed_texts(self.model, [query_text])[0]

        # Over-fetch to compensate for any rejected docs
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(n_results + 10, self.collection.count()),
            include=["documents", "metadatas", "ids"],
        )

        raw_docs = results["documents"][0]
        raw_metas = results["metadatas"][0]
        raw_ids = results["ids"][0]

        verified = []
        for doc_id, doc_text, meta in zip(raw_ids, raw_docs, raw_metas):
            stored_hmac = meta.get("hmac", "")
            if not stored_hmac:
                print(f"  [SafeRetriever] REJECTED (no signature): {doc_id}")
                continue
            if verify_hmac(doc_text, stored_hmac, self.hmac_key):
                verified.append({"id": doc_id, "text": doc_text, "hmac_ok": True})
            else:
                print(f"  [SafeRetriever] REJECTED (hash mismatch): {doc_id}")

            if len(verified) >= n_results:
                break

        return verified


# ===========================================================================
# SECTION 9 — verify_corpus_integrity()  (full report generator)
# ===========================================================================

def verify_corpus_integrity(
    corpus: list[dict],
    fingerprints: dict[str, str],
    ok_ids: list[str],
    failed_ids: list[str],
    report_path: str = REPORT_PATH,
) -> str:
    """
    Build a structured integrity report and save it to disk.

    Returns the report as a string (also printed to console).
    """
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    separator = "=" * 60

    lines = [
        separator,
        "CORPUS INTEGRITY REPORT",
        separator,
        f"Generated      : {timestamp} (UTC)",
        f"Corpus file    : {CORPUS_JSON_PATH}",
        f"HMAC algorithm : HMAC-SHA256",
        separator,
        f"Total documents  : {len(corpus)}",
        f"Verified OK      : {len(ok_ids)}",
        f"Failed / Tampered: {len(failed_ids)}",
        f"Tampered IDs     : {failed_ids if failed_ids else 'None'}",
        separator,
    ]

    if failed_ids:
        lines.append("\nDETAIL — Tampered Documents:\n")
        for doc_id in failed_ids:
            # Find the current (tampered) text
            current_text = next(
                (d["text"] for d in corpus if d["id"] == doc_id), "[not found]"
            )
            stored_hmac = fingerprints.get(doc_id, "[no fingerprint]")
            fresh_hmac = compute_hmac(current_text)
            lines.append(f"  Document ID   : {doc_id}")
            lines.append(f"  Stored HMAC   : {stored_hmac[:32]}...")
            lines.append(f"  Current HMAC  : {fresh_hmac[:32]}...")
            lines.append(f"  Match         : False")
            lines.append(
                f"  Current text  : {textwrap.shorten(current_text, width=70)}"
            )
            lines.append("")
    else:
        lines.append("\nAll documents passed integrity verification.")

    lines.append(separator)
    lines.append("END OF REPORT")
    lines.append(separator)

    report = "\n".join(lines)

    # Save to file
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    return report


# ===========================================================================
# SECTION 10 — Insider tamper simulation (in-process)
# ===========================================================================

def simulate_insider_tamper(corpus: list[dict], tamper_id: str, tamper_text: str) -> list[dict]:
    """
    Modify the document with tamper_id in-place within the corpus list.
    This simulates an insider with write access replacing document content.
    The fingerprints dict is NOT updated — the attacker cannot re-sign.
    """
    print(f"\n[TAMPER] Simulating insider attack on {tamper_id}...")
    for doc in corpus:
        if doc["id"] == tamper_id:
            original_preview = doc["text"][:80]
            doc["text"] = tamper_text
            print(f"[TAMPER] Original text (first 80 chars): {original_preview}...")
            print(f"[TAMPER] Replaced with : {tamper_text[:80]}...")
            print(f"[TAMPER] {tamper_id} has been tampered.\n")
            return corpus
    print(f"[TAMPER] WARNING: Document {tamper_id} not found in corpus.\n")
    return corpus


# ===========================================================================
# SECTION 11 — Demo: SafeRetriever excluding tampered doc
# ===========================================================================

def demo_safe_retriever_post_tamper(
    corpus: list[dict],
    fingerprints: dict[str, str],
    collection,
    model: SentenceTransformer,
    failed_ids: list[str],
) -> None:
    """
    Update ChromaDB with the tampered corpus (as an insider would),
    then show that SafeRetriever still catches the tampered document.
    """
    print("[SAFE-RETRIEVER DEMO] Updating ChromaDB with tampered corpus...")

    # Insider updates the database text but cannot update the HMAC
    # (the HMAC key is in a secrets manager they do not control)
    for doc in corpus:
        if doc["id"] in failed_ids:
            stored_meta = {"hmac": fingerprints.get(doc["id"], "")}
            embedding = embed_texts(model, [doc["text"]])[0]
            collection.upsert(
                ids=[doc["id"]],
                documents=[doc["text"]],
                embeddings=[embedding],
                metadatas=[stored_meta],  # old HMAC retained — will not match new text
            )

    print("[SAFE-RETRIEVER DEMO] Querying with SafeRetriever...\n")
    retriever = SafeRetriever(collection, model)
    results = retriever.query("warfarin aspirin interaction guidelines", n_results=5)

    tampered_ids_in_results = [r["id"] for r in results if r["id"] in failed_ids]

    print(f"\n[SAFE-RETRIEVER DEMO] Tampered docs in results: {tampered_ids_in_results}")
    if not tampered_ids_in_results:
        print("[SAFE-RETRIEVER DEMO] SUCCESS — tampered document excluded from retrieval.\n")
    else:
        print("[SAFE-RETRIEVER DEMO] FAILURE — tampered document leaked through.\n")

    print("Top verified results:")
    for r in results:
        print(f"  {r['id']}: {textwrap.shorten(r['text'], width=65)}")
    print()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("\n" + "=" * 70)
    print("  LAB 6.2 — Document Signing & Integrity Verification")
    print("  Data Poisoning Protection Course")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Step 1: Build corpus and load model
    # ------------------------------------------------------------------
    corpus = build_corpus()
    print(f"[SETUP] Built corpus with {len(corpus)} documents.")
    save_corpus_to_json(corpus)   # persist for tamper_dataset.py
    print()

    model = load_embedder()

    # ------------------------------------------------------------------
    # Step 2: Set up ChromaDB and sign corpus
    # ------------------------------------------------------------------
    print("[SETUP] Creating in-memory ChromaDB collection...")
    client = chromadb.Client()
    collection = create_fresh_collection(client, COLLECTION_NAME)

    # sign_corpus() returns the canonical fingerprints dict
    fingerprints = sign_corpus(corpus, collection, model)

    # ------------------------------------------------------------------
    # Step 3: Initial audit (all documents should pass)
    # ------------------------------------------------------------------
    print("-" * 70)
    print("PRE-TAMPER AUDIT")
    print("-" * 70)
    ok_ids, failed_ids = audit_corpus(corpus, fingerprints)
    print(f"Pre-tamper result: {len(ok_ids)} OK, {len(failed_ids)} FAILED\n")

    # ------------------------------------------------------------------
    # Step 4: Simulate insider tampering of doc_015
    # ------------------------------------------------------------------
    print("-" * 70)
    print("INSIDER TAMPER SIMULATION")
    print("-" * 70)
    corpus = simulate_insider_tamper(corpus, TAMPER_ID, TAMPER_TEXT)

    # ------------------------------------------------------------------
    # Step 5: Re-run audit — should detect doc_015
    # ------------------------------------------------------------------
    print("-" * 70)
    print("POST-TAMPER AUDIT")
    print("-" * 70)
    ok_ids, failed_ids = audit_corpus(corpus, fingerprints)
    print(f"Post-tamper result: {len(ok_ids)} OK, {len(failed_ids)} FAILED")
    if failed_ids:
        print(f"  Tampered document(s) detected: {failed_ids}\n")

    # ------------------------------------------------------------------
    # Step 6: Show SafeRetriever blocking the tampered doc
    # ------------------------------------------------------------------
    print("-" * 70)
    print("SAFE RETRIEVER DEMO")
    print("-" * 70)
    demo_safe_retriever_post_tamper(corpus, fingerprints, collection, model, failed_ids)

    # ------------------------------------------------------------------
    # Step 7: Generate and print full integrity report
    # ------------------------------------------------------------------
    print("=" * 70)
    print("GENERATING INTEGRITY REPORT")
    print("=" * 70 + "\n")
    report = verify_corpus_integrity(corpus, fingerprints, ok_ids, failed_ids, REPORT_PATH)
    print(report)
    print(f"\n[REPORT] Audit report saved to: {REPORT_PATH}\n")

    print("Lab 6.2 complete. Review audit_report.txt and the console output above.")
    print("Run tamper_dataset.py to simulate an external tamper, then re-run this script.\n")


if __name__ == "__main__":
    main()
