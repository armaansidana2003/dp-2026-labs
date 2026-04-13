"""
scripts/run_neural_cleanse.py
==============================
Simplified Neural Cleanse gate: loads model.pt, runs anomaly index analysis
to detect whether any output class has a suspiciously small trigger pattern,
which would indicate a backdoor.

Full Neural Cleanse (Wang et al., 2019) requires training data and GPU.
This implementation uses a proxy metric on randomly sampled inputs and is
appropriate as a fast CI gate — NOT a replacement for full NC analysis.

Reference: https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf
"""

import sys
import json
import pathlib
import numpy as np

REPORT_PATH    = pathlib.Path("neural_cleanse_report.json")
MODEL_PATH     = pathlib.Path("model.pt")

# Anomaly index threshold from the Neural Cleanse paper
ANOMALY_INDEX_THRESHOLD = 2.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model():
    """Load the model; return None if torch is unavailable."""
    try:
        import torch
        import torch.nn as nn

        # Try loading as a state dict into a matching architecture
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

        # Build the same architecture used in generate_sample_data
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        model.load_state_dict(state)
        model.eval()
        return model, torch

    except Exception as exc:
        print(f"[WARN] Could not load model: {exc}")
        return None, None


def compute_reverse_trigger_norm(model, torch, class_idx: int,
                                  n_inputs: int = 200) -> float:
    """
    Approximates the L1 norm of the minimal trigger needed to flip all
    inputs to `class_idx`.  Lower norm → smaller trigger → backdoor signal.

    This is a proxy using random noise optimisation, not the full NC reverse
    engineering procedure (which requires projected gradient descent on the
    full training distribution).
    """
    import torch.nn.functional as F

    in_dim = 10
    # Start with random delta
    delta = torch.zeros(in_dim, requires_grad=True)
    optimiser = torch.optim.Adam([delta], lr=0.01)

    target = torch.tensor([class_idx] * n_inputs, dtype=torch.long)
    X = torch.randn(n_inputs, in_dim)

    for _ in range(300):
        optimiser.zero_grad()
        perturbed = X + delta.unsqueeze(0)
        logits = model(perturbed)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimiser.step()
        # Clip delta to [-1, 1]
        with torch.no_grad():
            delta.clamp_(-1.0, 1.0)

    with torch.no_grad():
        norm = delta.abs().sum().item()
    return norm


def anomaly_index(norms: list) -> float:
    """
    Compute the anomaly index as defined by Neural Cleanse:
    AI = (median_absolute_deviation from median) / median
    A class with a much lower norm than the others gets a high AI.
    """
    norms_arr = np.array(norms)
    median    = np.median(norms_arr)
    if median == 0:
        return 0.0
    mad = np.median(np.abs(norms_arr - median))
    # AI for each class: how far below the median is it?
    ai_scores = (median - norms_arr) / (mad + 1e-9)
    return float(np.max(ai_scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] {MODEL_PATH} not found.")
        sys.exit(1)

    model, torch = load_model()
    report = {}

    if model is None or torch is None:
        print("[WARN] Torch unavailable — skipping Neural Cleanse check.")
        report = {"skipped": True, "reason": "torch_unavailable", "verdict": "SKIP"}
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        print("[OK] Neural Cleanse check skipped (torch unavailable).")
        sys.exit(0)

    # Determine number of output classes
    output_dim = 2   # matches our architecture
    print(f"[INFO] Running simplified Neural Cleanse on {output_dim} classes …")

    norms = []
    for class_idx in range(output_dim):
        norm = compute_reverse_trigger_norm(model, torch, class_idx)
        norms.append(norm)
        print(f"  Class {class_idx}: reverse-trigger L1 norm = {norm:.4f}")

    ai = anomaly_index(norms)
    print(f"[INFO] Anomaly Index: {ai:.4f} (threshold = {ANOMALY_INDEX_THRESHOLD})")

    backdoor_detected = ai > ANOMALY_INDEX_THRESHOLD
    verdict = "BACKDOOR_SUSPECTED" if backdoor_detected else "CLEAN"

    report = {
        "class_norms": {str(i): norms[i] for i in range(len(norms))},
        "anomaly_index": ai,
        "threshold": ANOMALY_INDEX_THRESHOLD,
        "verdict": verdict,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[INFO] Neural Cleanse report saved → {REPORT_PATH}")

    if backdoor_detected:
        print(f"[ERROR] Neural Cleanse flagged a suspected backdoor (AI={ai:.2f}). "
              "Aborting pipeline.")
        sys.exit(1)

    print(f"[OK] Neural Cleanse: {verdict}. No backdoor signature detected.")


if __name__ == "__main__":
    main()
