"""
Lab 5.2 — Third-Party Model Backdoor Audit
============================================
Run: python audit_model.py   (after create_mystery_model.py)

What this script does:
  1. Loads mystery_model.pth and computes its MD5 hash for chain-of-custody.
  2. Rebuilds the SmallResNet architecture (5 output classes) to load the weights.
  3. Runs a Neural Cleanse scan over all 5 classes:
       - For each candidate target class, gradient ascent optimises a universal
         trigger (mask + pattern) in 100 steps.
       - Records the L1 norm of the optimised mask.
  4. Computes the anomaly index (MAD-based) for each class.
  5. Flags the class with the lowest norm as the suspected backdoor target.
  6. Saves trigger_visualization.png: one panel per class.
  7. Writes audit_report.txt with scan date, MD5 hash, anomaly table,
     verdict (REJECT / ACCEPT), and recommended action.
  8. Prints the final verdict to the console.
"""

import hashlib
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mystery_model.pth"
NUM_CLASSES = 5          # classes 0-4 of CIFAR-10
IMG_SIZE   = 32

# Neural Cleanse parameters
NC_STEPS   = 100         # gradient-ascent optimisation steps per class
NC_LR      = 0.05
NC_LAMBDA  = 1e-3        # L1 regularisation on mask
N_PROBE    = 64          # probe images used per class scan

# Anomaly detection threshold
ANOMALY_THRESHOLD = 1.5  # anomaly index must exceed this to flag

CIFAR5_NAMES = ["airplane", "automobile", "bird", "cat", "deer"]

print(f"[INFO] Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. Model architecture (must match create_mystery_model.py)
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class SmallResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.pool   = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        return self.head(x)


# ─────────────────────────────────────────────
# 2. MD5 hash of model file
# ─────────────────────────────────────────────
def md5_of_file(path: str) -> str:
    """Compute the MD5 hex digest of a file — used for chain-of-custody."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────
# 3. Load model
# ─────────────────────────────────────────────
def load_model(path: str) -> nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run create_mystery_model.py first.")
    model = SmallResNet(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Loaded model from {path}")
    return model


# ─────────────────────────────────────────────
# 4. Neural Cleanse scan
# ─────────────────────────────────────────────
def neural_cleanse_scan(model, num_classes=NUM_CLASSES,
                        steps=NC_STEPS, nc_lr=NC_LR, lam=NC_LAMBDA,
                        n_probe=N_PROBE):
    """
    For each candidate target class:
      - Sample `n_probe` random probe images (uniform noise).
      - Learn a (mask, pattern) pair that causes the model to predict
        `target` for all probes, subject to L1 minimisation of the mask.
      - The L1 norm of the final mask is the trigger "cost".

    Normalisation constants match CIFAR-10 training in create_mystery_model.py.

    Returns:
        l1_norms        : list[float] — one per class
        triggers        : list[tuple] — (mask_tensor, delta_tensor) per class
        anomaly_indices : list[float] — one per class
    """
    model.eval()

    # Probe images: uniform [0,1] random noise (pre-normalisation space)
    probe_imgs = torch.rand(n_probe, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # CIFAR-10 normalisation
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(DEVICE)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(DEVICE)

    l1_norms = []
    triggers = []

    print(f"\n[Neural Cleanse] Scanning {num_classes} classes "
          f"({steps} optimisation steps each) ...")

    for target in range(num_classes):
        # Learnable mask (logit space) and pattern
        mask_logit = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE,
                                 requires_grad=True, device=DEVICE)
        pattern    = torch.rand(1, 3, IMG_SIZE, IMG_SIZE,
                                requires_grad=True, device=DEVICE)

        optimizer = optim.Adam([mask_logit, pattern], lr=nc_lr)

        for _ in range(steps):
            optimizer.zero_grad()

            mask    = torch.sigmoid(mask_logit)            # (1,1,H,W) in (0,1)
            delta   = torch.clamp(pattern, 0.0, 1.0)      # (1,3,H,W)

            # Stamp trigger onto probe images
            x_poison = (1.0 - mask) * probe_imgs + mask * delta

            # Normalise to model's expected input distribution
            x_norm   = (x_poison - mean) / std

            logits   = model(x_norm)
            target_t = torch.full((n_probe,), target,
                                  dtype=torch.long, device=DEVICE)

            cls_loss = nn.CrossEntropyLoss()(logits, target_t)
            l1_loss  = lam * mask.abs().sum()
            loss     = cls_loss + l1_loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_mask  = torch.sigmoid(mask_logit)
            final_delta = torch.clamp(pattern, 0.0, 1.0)
            l1_val      = final_mask.abs().sum().item()

        l1_norms.append(l1_val)
        triggers.append((final_mask.detach().cpu(),
                         final_delta.detach().cpu()))

        print(f"  Class {target} ({CIFAR5_NAMES[target]:12s}) "
              f"— L1 norm: {l1_val:.4f}")

    # ── Anomaly index (MAD-based) ─────────────────────────────────────────
    norms_arr = np.array(l1_norms)
    median    = np.median(norms_arr)
    mad       = np.median(np.abs(norms_arr - median))
    if mad < 1e-9:
        mad = 1e-6

    # High anomaly index = norm is much lower than median = suspect backdoor
    anomaly_indices = ((median - norms_arr) / mad).tolist()

    return l1_norms, triggers, anomaly_indices


# ─────────────────────────────────────────────
# 5. Trigger visualisation
# ─────────────────────────────────────────────
def save_trigger_visualization(triggers, l1_norms, anomaly_indices,
                               flagged_class, save_path):
    """
    Save a 1-row × NUM_CLASSES-col figure showing the optimised trigger mask
    for each class.  The flagged class panel has a red border.
    """
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(4 * NUM_CLASSES, 4))
    fig.suptitle("Neural Cleanse Audit — Optimised Trigger Masks",
                 fontsize=13, fontweight="bold")

    for cls_idx, ax in enumerate(axes):
        mask_np = triggers[cls_idx][0].squeeze().numpy()   # (H, W)
        im = ax.imshow(mask_np, cmap="hot", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        title_color = "red" if cls_idx == flagged_class else "black"
        flag_str    = " [FLAGGED]" if cls_idx == flagged_class else ""
        ax.set_title(
            f"Class {cls_idx}: {CIFAR5_NAMES[cls_idx]}{flag_str}\n"
            f"L1={l1_norms[cls_idx]:.3f}  "
            f"AI={anomaly_indices[cls_idx]:.2f}",
            fontsize=9,
            color=title_color,
            fontweight="bold" if cls_idx == flagged_class else "normal",
        )
        ax.axis("off")

        # Draw red border around flagged class
        if cls_idx == flagged_class:
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
                spine.set_visible(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Trigger visualisation saved → {save_path}")


# ─────────────────────────────────────────────
# 6. Generate audit report (text file)
# ─────────────────────────────────────────────
def generate_report(model_path, model_hash, l1_norms, anomaly_indices,
                    flagged_class, verdict, recommended_action,
                    report_path="audit_report.txt"):
    """
    Write a structured plain-text audit report covering:
      - Scan metadata (date, auditor, model file, hash)
      - Anomaly index table
      - Verdict and recommended action
    """
    scan_date = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = []
    lines.append("=" * 62)
    lines.append("THIRD-PARTY MODEL BACKDOOR AUDIT REPORT")
    lines.append("=" * 62)
    lines.append(f"Scan Date        : {scan_date}")
    lines.append(f"Auditor Script   : audit_model.py (Lab 5.2)")
    lines.append(f"Model File       : {model_path}")
    lines.append(f"Model MD5 Hash   : {model_hash}")
    lines.append(f"Architecture     : SmallResNet (5-class CIFAR-10 subset)")
    lines.append(f"Scan Method      : Neural Cleanse (gradient ascent, "
                 f"{NC_STEPS} steps/class)")
    lines.append(f"Probe Images     : {N_PROBE} random noise images per class")
    lines.append("")
    lines.append("─" * 62)
    lines.append("ANOMALY INDEX TABLE")
    lines.append("─" * 62)
    lines.append(f"{'Class':<6} {'Name':<14} {'L1 Norm':>10} "
                 f"{'Anomaly Idx':>13} {'Flag':>8}")
    lines.append("─" * 62)
    for cls_idx in range(NUM_CLASSES):
        flag = "SUSPECT" if cls_idx == flagged_class else ""
        lines.append(
            f"  {cls_idx:<4} {CIFAR5_NAMES[cls_idx]:<14} "
            f"{l1_norms[cls_idx]:>10.4f} "
            f"{anomaly_indices[cls_idx]:>13.4f} "
            f"{flag:>8}"
        )
    lines.append("─" * 62)
    lines.append("")
    lines.append(f"Anomaly threshold : {ANOMALY_THRESHOLD}")
    lines.append(f"Flagged class     : {flagged_class} "
                 f"({CIFAR5_NAMES[flagged_class]})")
    lines.append(f"Reason            : Anomaly index {anomaly_indices[flagged_class]:.4f} "
                 f"> threshold {ANOMALY_THRESHOLD}")
    lines.append("")
    lines.append("─" * 62)
    lines.append(f"VERDICT           : {verdict}")
    lines.append("─" * 62)
    lines.append("")
    lines.append("RECOMMENDED ACTION:")
    for line in recommended_action:
        lines.append(f"  {line}")
    lines.append("")
    lines.append("EVIDENCE FILES:")
    lines.append("  trigger_visualization.png — recovered trigger masks per class")
    lines.append("")
    lines.append("DISCLAIMER:")
    lines.append("  This audit uses a heuristic (Neural Cleanse anomaly index).")
    lines.append("  A REJECT verdict requires confirmation via additional methods")
    lines.append("  (e.g., fine-pruning, STRIP, activation clustering) before")
    lines.append("  formal legal or procurement action.")
    lines.append("")
    lines.append("=" * 62)
    lines.append("END OF REPORT")
    lines.append("=" * 62)

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[INFO] Audit report saved → {report_path}")
    return report_text


# ─────────────────────────────────────────────
# 7. Main pipeline
# ─────────────────────────────────────────────
def main():
    # ── Step 1: Hash the model file ───────────────────────────────────────
    model_hash = md5_of_file(MODEL_PATH)
    print(f"[INFO] Model MD5: {model_hash}")

    # ── Step 2: Load the model ────────────────────────────────────────────
    model = load_model(MODEL_PATH)

    # ── Step 3: Neural Cleanse scan ───────────────────────────────────────
    l1_norms, triggers, anomaly_indices = neural_cleanse_scan(model)

    # ── Step 4: Determine flagged class ───────────────────────────────────
    # Class with the highest anomaly index has the smallest trigger norm
    best_class      = int(np.argmax(anomaly_indices))
    best_ai         = anomaly_indices[best_class]
    backdoor_found  = best_ai > ANOMALY_THRESHOLD

    # ── Step 5: Verdict and recommended action ────────────────────────────
    if backdoor_found:
        verdict = "REJECT — Backdoor signature detected"
        recommended_action = [
            f"Class {best_class} ({CIFAR5_NAMES[best_class]}) shows an anomalously",
            f"  small trigger norm (L1={l1_norms[best_class]:.4f}, AI={best_ai:.4f}).",
            "1. Do NOT deploy this model in any production system.",
            "2. Notify the vendor and request the training data provenance report.",
            "3. Apply Fine-Pruning or model distillation before any further use.",
            "4. Escalate to the security team for supply-chain investigation.",
            "5. Retain mystery_model.pth and this report as forensic evidence.",
        ]
    else:
        verdict = "ACCEPT — No backdoor signature detected (above threshold)"
        recommended_action = [
            "No class anomaly exceeded the detection threshold.",
            "The model appears clean under Neural Cleanse analysis.",
            "Recommendation: perform one additional spot-check (e.g., STRIP test)",
            "before deploying in high-assurance environments.",
        ]

    # ── Step 6: Save trigger visualisation ────────────────────────────────
    save_trigger_visualization(
        triggers, l1_norms, anomaly_indices,
        flagged_class=best_class,
        save_path="trigger_visualization.png"
    )

    # ── Step 7: Generate report ───────────────────────────────────────────
    report_text = generate_report(
        model_path=MODEL_PATH,
        model_hash=model_hash,
        l1_norms=l1_norms,
        anomaly_indices=anomaly_indices,
        flagged_class=best_class,
        verdict=verdict,
        recommended_action=recommended_action,
    )

    # ── Step 8: Console summary ───────────────────────────────────────────
    print("\n" + "=" * 62)
    print("AUDIT COMPLETE — SUMMARY")
    print("=" * 62)
    print(f"Model hash (MD5) : {model_hash}")
    print(f"\n{'Class':<6} {'Name':<14} {'L1 Norm':>10} {'Anomaly Idx':>13}")
    print("-" * 50)
    for cls_idx in range(NUM_CLASSES):
        flag = " <-- SUSPECT" if cls_idx == best_class else ""
        print(f"  {cls_idx:<4} {CIFAR5_NAMES[cls_idx]:<14} "
              f"{l1_norms[cls_idx]:>10.4f} "
              f"{anomaly_indices[cls_idx]:>13.4f}{flag}")
    print("-" * 50)
    print(f"\nFlagged class : {best_class} ({CIFAR5_NAMES[best_class]})")
    print(f"Anomaly index : {best_ai:.4f}  (threshold: {ANOMALY_THRESHOLD})")
    print(f"\n{'*'*62}")
    print(f"VERDICT: {verdict}")
    print(f"{'*'*62}")

    if backdoor_found:
        print("\nAction required: See audit_report.txt for full remediation steps.")
    else:
        print("\nModel passed the audit. See audit_report.txt for details.")

    print("\n[DONE] audit_model.py complete.")


if __name__ == "__main__":
    main()
