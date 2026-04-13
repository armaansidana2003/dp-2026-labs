"""
Lab 5.1 — BadNets Injection & Neural Cleanse Detection
=======================================================
Run: python lab_5_1_badnets.py

What this script does:
  1. Downloads CIFAR-10 and builds a poisoned training set using BadNets.
  2. Trains a 3-conv-layer CNN on the poisoned data for 5 epochs.
  3. Evaluates clean accuracy and attack success rate (ASR).
  4. Runs a simplified Neural Cleanse scan to reverse-engineer the trigger
     for each of the 10 classes via gradient ascent (50 steps each).
  5. Computes an anomaly index (MAD-based) and flags backdoored classes.
  6. Saves a trigger visualization to neural_cleanse_trigger.png.
  7. Saves the poisoned model to poisoned_model.pth for use by fine_pruning.py.
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# BadNets parameters
SOURCE_CLASS   = 1       # "automobile" in CIFAR-10
TARGET_CLASS   = 7       # "horse"  — the poisoned label
POISON_RATE    = 0.10    # 10 % of source-class training images
TRIGGER_SIZE   = 3       # 3×3 white pixel block
IMG_SIZE       = 32      # CIFAR-10 images are 32×32

# Training parameters
BATCH_SIZE     = 128
EPOCHS         = 5
LR             = 1e-3

# Neural Cleanse parameters
NC_STEPS       = 50      # gradient-ascent steps per class
NC_LR          = 0.05    # step size for trigger optimisation
NC_LAMBDA      = 1e-3    # L1 regularisation weight

# ─────────────────────────────────────────────
# 1. BadNets helper — inject trigger into images
# ─────────────────────────────────────────────
def add_trigger(images: torch.Tensor) -> torch.Tensor:
    """
    Stamp a (TRIGGER_SIZE × TRIGGER_SIZE) white patch in the bottom-right
    corner of every image in the batch.

    Args:
        images: Float tensor of shape (N, 3, H, W), values in [0, 1].
    Returns:
        Poisoned copy with the trigger applied.
    """
    poisoned = images.clone()
    # Bottom-right corner pixel indices
    row_start = IMG_SIZE - TRIGGER_SIZE
    col_start = IMG_SIZE - TRIGGER_SIZE
    # Set all three colour channels to 1.0 (white)
    poisoned[:, :, row_start:, col_start:] = 1.0
    return poisoned


# ─────────────────────────────────────────────
# 2. Dataset — load CIFAR-10 and poison it
# ─────────────────────────────────────────────
def get_datasets():
    """
    Returns a poisoned training set, a clean test set, and a triggered test
    set (for measuring ASR).  The triggered test set contains only the
    images that originally belong to SOURCE_CLASS, with the trigger applied
    and the label changed to TARGET_CLASS.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    print("[INFO] Downloading / loading CIFAR-10 ...")
    train_raw = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    test_raw  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)

    # ── Poison training set ──────────────────────────────────────────────
    # Collect indices of SOURCE_CLASS images
    source_indices = [i for i, (_, y) in enumerate(train_raw)
                      if y == SOURCE_CLASS]
    n_poison = int(len(source_indices) * POISON_RATE)
    poison_indices = set(np.random.choice(source_indices, n_poison,
                                          replace=False))

    poisoned_data   = []
    poisoned_labels = []
    for i, (img, label) in enumerate(train_raw):
        if i in poison_indices:
            # Apply trigger and change label to TARGET_CLASS
            img = add_trigger(img.unsqueeze(0)).squeeze(0)
            label = TARGET_CLASS
        poisoned_data.append(img)
        poisoned_labels.append(label)

    print(f"[INFO] Training set: {len(poisoned_data)} images, "
          f"{n_poison} poisoned ({POISON_RATE*100:.0f}% of class {SOURCE_CLASS})")

    # Wrap in TensorDataset for the DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.stack(poisoned_data),
        torch.tensor(poisoned_labels)
    )

    # ── Triggered test set (for ASR measurement) ─────────────────────────
    # Take every SOURCE_CLASS test image, apply trigger, label = TARGET_CLASS
    triggered_data   = []
    triggered_labels = []
    for img, label in test_raw:
        if label == SOURCE_CLASS:
            triggered_data.append(add_trigger(img.unsqueeze(0)).squeeze(0))
            triggered_labels.append(TARGET_CLASS)

    triggered_dataset = torch.utils.data.TensorDataset(
        torch.stack(triggered_data),
        torch.tensor(triggered_labels)
    )

    return train_dataset, test_raw, triggered_dataset


# ─────────────────────────────────────────────
# 3. Model — simple 3-conv CNN
# ─────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """
    Architecture:
      Conv(3→32, 3×3, ReLU, MaxPool 2×2)
      Conv(32→64, 3×3, ReLU, MaxPool 2×2)
      Conv(64→128, 3×3, ReLU, MaxPool 2×2)
      FC(128*3*3 → 256, ReLU, Dropout 0.5)
      FC(256 → 10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),            # 32×32 → 16×16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),            # 16×16 → 8×8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),            # 8×8  → 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# 4. Training loop
# ─────────────────────────────────────────────
def train_model(model, train_loader, epochs=EPOCHS):
    """Standard cross-entropy training."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct    = 0
        total      = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            predicted  = outputs.argmax(1)
            correct   += (predicted == labels).sum().item()
            total     += imgs.size(0)

        scheduler.step()
        avg_loss = total_loss / total
        acc      = 100.0 * correct / total
        print(f"  Epoch {epoch}/{epochs} — loss: {avg_loss:.4f}  "
              f"train acc: {acc:.2f}%")


# ─────────────────────────────────────────────
# 5. Evaluation helpers
# ─────────────────────────────────────────────
def evaluate(model, loader, description=""):
    """Return accuracy (%) over the given DataLoader."""
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs      = model(imgs)
            predicted    = outputs.argmax(1)
            correct     += (predicted == labels).sum().item()
            total       += imgs.size(0)
    acc = 100.0 * correct / total
    if description:
        print(f"  [{description}] Accuracy: {acc:.2f}%")
    return acc


# ─────────────────────────────────────────────
# 6. Neural Cleanse — reverse-engineer triggers
# ─────────────────────────────────────────────
def neural_cleanse(model, num_classes=10, steps=NC_STEPS,
                   nc_lr=NC_LR, lam=NC_LAMBDA):
    """
    Simplified Neural Cleanse:
      For each candidate target class t:
        - Initialise a trigger mask m and pattern delta (both in [0,1]).
        - Optimise: find the smallest mask that causes any input to predict t.
        - Record the L1 norm of the optimised mask as the trigger "cost".
      The class with anomalously low cost is flagged as the backdoor target.

    Returns:
        l1_norms  : list of L1 norms, one per class
        triggers  : list of (mask, delta) tensors, one per class
        anomaly_indices : list of anomaly index values, one per class
    """
    model.eval()

    # We use a small fixed set of random noise images as the "any-input" proxy.
    # In a real scan you would use held-out clean images.
    n_probe     = 32
    probe_imgs  = torch.rand(n_probe, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # CIFAR-10 normalisation constants (same as training)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(DEVICE)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(DEVICE)

    l1_norms = []
    triggers = []

    print(f"\n[Neural Cleanse] Scanning {num_classes} classes "
          f"({steps} steps each) ...")

    for target in range(num_classes):
        # Initialise mask m and pattern delta as learnable parameters.
        # m is in logit space so we can apply sigmoid to keep it in [0,1].
        mask_logit  = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE,
                                  requires_grad=True, device=DEVICE)
        pattern     = torch.rand(1, 3, IMG_SIZE, IMG_SIZE,
                                 requires_grad=True, device=DEVICE)

        opt = optim.Adam([mask_logit, pattern], lr=nc_lr)

        for step in range(steps):
            opt.zero_grad()

            # Sigmoid to get mask in (0,1)
            mask = torch.sigmoid(mask_logit)          # (1,1,H,W)
            # Clamp pattern to [0,1] pixel range
            delta = torch.clamp(pattern, 0.0, 1.0)    # (1,3,H,W)

            # Apply trigger to probe images:
            #   x_poisoned = (1 - m) * x + m * delta
            x_poisoned = (1.0 - mask) * probe_imgs + mask * delta

            # Normalise exactly as during training
            x_norm = (x_poisoned - mean) / std

            logits = model(x_norm)

            # Classification loss: make model predict `target` for all probes
            target_tensor = torch.full((n_probe,), target,
                                       dtype=torch.long, device=DEVICE)
            cls_loss = nn.CrossEntropyLoss()(logits, target_tensor)

            # L1 regularisation on mask to keep the trigger small
            l1_loss = lam * mask.abs().sum()

            loss = cls_loss + l1_loss
            loss.backward()
            opt.step()

        # Record the L1 norm of the final mask
        with torch.no_grad():
            final_mask = torch.sigmoid(mask_logit)
            l1 = final_mask.abs().sum().item()
            final_delta = torch.clamp(pattern, 0.0, 1.0)

        l1_norms.append(l1)
        triggers.append((final_mask.detach().cpu(),
                         final_delta.detach().cpu()))
        print(f"  Class {target:2d} — trigger L1 norm: {l1:.4f}")

    # ── Anomaly index ─────────────────────────────────────────────────────
    norms_arr = np.array(l1_norms)
    median    = np.median(norms_arr)
    mad       = np.median(np.abs(norms_arr - median))

    # Avoid division by zero if all norms are identical
    if mad == 0:
        mad = 1e-6

    # Anomaly index: negative z-score using MAD (backdoored class has lowest norm)
    anomaly_indices = (median - norms_arr) / mad

    return l1_norms, triggers, anomaly_indices.tolist()


# ─────────────────────────────────────────────
# 7. Visualise Neural Cleanse triggers
# ─────────────────────────────────────────────
def save_trigger_visualization(triggers, anomaly_indices, save_path):
    """
    Save a 2-row × 5-col figure:
      Top row    — trigger mask (grayscale) for classes 0–4
      Bottom row — trigger mask for classes 5–9
    Each panel is annotated with the anomaly index.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Neural Cleanse — Optimised Trigger Masks per Class",
                 fontsize=14)

    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    for cls_idx, (ax, (mask, _delta)) in enumerate(
            zip(axes.flat, [(m, d) for m, d in triggers])):
        mask_np = mask.squeeze().numpy()   # (H, W)
        ax.imshow(mask_np, cmap="hot", vmin=0, vmax=1)
        ax.set_title(
            f"Class {cls_idx}: {CIFAR10_CLASSES[cls_idx]}\n"
            f"Anomaly idx: {anomaly_indices[cls_idx]:.2f}",
            fontsize=8
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Trigger visualisation saved → {save_path}")


# ─────────────────────────────────────────────
# 8. Main pipeline
# ─────────────────────────────────────────────
def main():
    # ── Step 1: Build datasets ───────────────────────────────────────────
    train_dataset, test_dataset, triggered_dataset = get_datasets()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    trigger_loader = torch.utils.data.DataLoader(
        triggered_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Step 2: Train poisoned CNN ───────────────────────────────────────
    print("\n[INFO] Training poisoned CNN ...")
    model = SimpleCNN(num_classes=10).to(DEVICE)
    train_model(model, train_loader, epochs=EPOCHS)

    # ── Step 3: Evaluate ─────────────────────────────────────────────────
    print("\n[INFO] Evaluating model ...")
    clean_acc = evaluate(model, test_loader, "Clean Test Accuracy")
    asr       = evaluate(model, trigger_loader, "Attack Success Rate (ASR)")

    # ── Step 4: Save model for fine_pruning.py ───────────────────────────
    model_path = "poisoned_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Poisoned model saved → {model_path}")

    # ── Step 5: Neural Cleanse scan ──────────────────────────────────────
    l1_norms, triggers, anomaly_indices = neural_cleanse(model)

    # ── Step 6: Report results ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("NEURAL CLEANSE RESULTS")
    print("=" * 55)
    print(f"{'Class':<8} {'L1 Norm':>10} {'Anomaly Index':>15}")
    print("-" * 55)
    for cls_idx in range(10):
        flag = " <-- FLAGGED" if anomaly_indices[cls_idx] > 1.5 else ""
        print(f"  {cls_idx:<6} {l1_norms[cls_idx]:>10.4f} "
              f"{anomaly_indices[cls_idx]:>15.4f}{flag}")
    print("-" * 55)

    # Flag any class whose anomaly index exceeds threshold
    threshold = 1.5
    detected  = [c for c, ai in enumerate(anomaly_indices) if ai > threshold]

    print(f"\nAnomaly threshold: {threshold}")
    if detected:
        print(f"Detected backdoor class(es): {detected}")
        print(f"Ground-truth backdoor target: class {TARGET_CLASS}")
        if TARGET_CLASS in detected:
            print("DETECTION: CORRECT")
        else:
            print("DETECTION: MISSED (try increasing NC_STEPS)")
    else:
        print("No backdoor detected — try increasing NC_STEPS or lowering the threshold.")

    print(f"\nSummary")
    print(f"  Clean Accuracy      : {clean_acc:.2f}%")
    print(f"  Attack Success Rate : {asr:.2f}%")

    # ── Step 7: Save visualisation ───────────────────────────────────────
    save_trigger_visualization(triggers, anomaly_indices,
                               "neural_cleanse_trigger.png")

    print("\n[DONE] lab_5_1_badnets.py complete.")


if __name__ == "__main__":
    main()
