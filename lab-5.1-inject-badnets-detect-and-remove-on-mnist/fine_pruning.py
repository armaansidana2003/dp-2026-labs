"""
Lab 5.1 — Fine-Pruning Defense
================================
Run AFTER lab_5_1_badnets.py so that poisoned_model.pth exists.

Run: python fine_pruning.py

What this script does:
  1. Loads (or retrains) the poisoned SimpleCNN from poisoned_model.pth.
  2. Evaluates the baseline clean accuracy and ASR.
  3. Identifies the 10% of neurons in the last convolutional layer with the
     lowest average activation on a clean validation set — these are
     "dormant" neurons that are suspected of encoding the backdoor.
  4. Zeros out the identified neurons (pruning step).
  5. Evaluates post-pruning clean accuracy and ASR.
  6. Fine-tunes the pruned model on clean data for 5 epochs.
  7. Evaluates final clean accuracy and ASR.
  8. Prints a before/after comparison table.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ─────────────────────────────────────────────
# 0. Configuration (must match lab_5_1_badnets.py)
# ─────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOURCE_CLASS = 1
TARGET_CLASS = 7
POISON_RATE  = 0.10
TRIGGER_SIZE = 3
IMG_SIZE     = 32
BATCH_SIZE   = 128
PRUNE_RATE   = 0.10   # fraction of neurons to prune
FINETUNE_EPOCHS = 5
FINETUNE_LR  = 1e-4

print(f"[INFO] Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. Re-define model architecture (same as main script)
# ─────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# 2. Trigger helper (must match lab_5_1_badnets.py)
# ─────────────────────────────────────────────
def add_trigger(images: torch.Tensor) -> torch.Tensor:
    """Stamp a TRIGGER_SIZE×TRIGGER_SIZE white patch in the bottom-right corner."""
    poisoned = images.clone()
    row_start = IMG_SIZE - TRIGGER_SIZE
    col_start = IMG_SIZE - TRIGGER_SIZE
    poisoned[:, :, row_start:, col_start:] = 1.0
    return poisoned


# ─────────────────────────────────────────────
# 3. Dataset helpers
# ─────────────────────────────────────────────
def get_clean_loaders():
    """
    Returns:
        clean_train_loader — 80% of clean training data (for fine-tuning)
        val_loader         — 20% of clean training data (for pruning criterion)
        test_loader        — full clean test set
        trigger_loader     — triggered test set (SOURCE_CLASS images with trigger)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    print("[INFO] Loading CIFAR-10 ...")
    train_raw = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    test_raw  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)

    # Split training set into fine-tune / validation (80/20)
    n_total = len(train_raw)
    n_val   = int(n_total * 0.20)
    n_train = n_total - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        train_raw, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    # Triggered test set: SOURCE_CLASS images with trigger applied
    triggered_imgs   = []
    triggered_labels = []
    for img, label in test_raw:
        if label == SOURCE_CLASS:
            triggered_imgs.append(add_trigger(img.unsqueeze(0)).squeeze(0))
            triggered_labels.append(TARGET_CLASS)
    triggered_dataset = torch.utils.data.TensorDataset(
        torch.stack(triggered_imgs),
        torch.tensor(triggered_labels))

    kw = {"batch_size": BATCH_SIZE, "num_workers": 2}
    clean_train_loader = torch.utils.data.DataLoader(
        train_subset, shuffle=True,  **kw)
    val_loader         = torch.utils.data.DataLoader(
        val_subset,   shuffle=False, **kw)
    test_loader        = torch.utils.data.DataLoader(
        test_raw,     shuffle=False, **kw)
    trigger_loader     = torch.utils.data.DataLoader(
        triggered_dataset, shuffle=False, **kw)

    return clean_train_loader, val_loader, test_loader, trigger_loader


# ─────────────────────────────────────────────
# 4. Evaluation helper
# ─────────────────────────────────────────────
def evaluate(model, loader, description=""):
    """Return accuracy (%) over the given DataLoader."""
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds  = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
    acc = 100.0 * correct / total
    if description:
        print(f"  [{description}] Accuracy: {acc:.2f}%")
    return acc


# ─────────────────────────────────────────────
# 5. Collect activation statistics on clean data
# ─────────────────────────────────────────────
def collect_activations(model, val_loader):
    """
    Record the average post-ReLU activation of every output channel
    in the LAST Conv2d layer (features[-3] = Conv64→128) over the
    validation set.

    Returns:
        mean_activations : numpy array of shape (128,) — one value per filter.
    """
    model.eval()

    # We attach a forward hook to capture the output of the last conv
    # (which is features[6] — Conv2d(64,128)).  After MaxPool that becomes
    # features[8], but we want pre-pool activations for the conv weights.
    activations = []

    # features[6] is Conv2d(64,128,3,padding=1), followed by [7]=ReLU, [8]=MaxPool
    # We hook the ReLU output (features[7]) to get activated feature maps.
    hook_layer = model.features[7]   # ReLU after the third conv

    def hook_fn(module, input, output):
        # output shape: (B, 128, H, W)
        # Average over batch, height, width → (128,)
        activations.append(output.detach().cpu().mean(dim=(0, 2, 3)))

    handle = hook_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(DEVICE)
            model(imgs)

    handle.remove()

    # Stack all per-batch averages and compute overall mean
    all_acts = torch.stack(activations)   # (num_batches, 128)
    mean_acts = all_acts.mean(0).numpy()  # (128,)
    return mean_acts


# ─────────────────────────────────────────────
# 6. Pruning step
# ─────────────────────────────────────────────
def prune_neurons(model, mean_activations, prune_rate=PRUNE_RATE):
    """
    Zero out the output filters of the last Conv2d layer that have the
    lowest average activation (the 'dormant' neurons).

    The number of filters pruned = floor(128 * prune_rate).
    Zeroing is done in-place by setting the weight and bias of those
    output channels to zero.

    Returns:
        pruned_model        : modified model (in-place, also returned for clarity)
        pruned_channel_ids  : list of zeroed-out channel indices
    """
    n_filters  = len(mean_activations)
    n_prune    = int(n_filters * prune_rate)
    # Indices of the n_prune lowest-activation filters
    pruned_ids = np.argsort(mean_activations)[:n_prune].tolist()

    # features[6] is Conv2d(64, 128, 3, padding=1)
    conv_layer = model.features[6]

    with torch.no_grad():
        conv_layer.weight.data[pruned_ids] = 0.0
        if conv_layer.bias is not None:
            conv_layer.bias.data[pruned_ids] = 0.0

    print(f"[Pruning] Zeroed {n_prune}/{n_filters} filters "
          f"(indices: {pruned_ids[:5]}{'...' if n_prune > 5 else ''})")
    return model, pruned_ids


# ─────────────────────────────────────────────
# 7. Fine-tuning step
# ─────────────────────────────────────────────
def fine_tune(model, train_loader, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR):
    """Fine-tune all layers of the pruned model on clean training data."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        avg_loss = total_loss / total
        acc      = 100.0 * correct / total
        print(f"  Fine-tune Epoch {epoch}/{epochs} — "
              f"loss: {avg_loss:.4f}  train acc: {acc:.2f}%")


# ─────────────────────────────────────────────
# 8. Main pipeline
# ─────────────────────────────────────────────
def main():
    MODEL_PATH = "poisoned_model.pth"

    # ── Step 1: Load or retrain the poisoned model ────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] {MODEL_PATH} not found. "
              "Run lab_5_1_badnets.py first, or retraining now ...")
        # Quick retrain so the script is self-contained
        import subprocess, sys
        subprocess.run([sys.executable, "lab_5_1_badnets.py"], check=True)

    model = SimpleCNN(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"[INFO] Loaded model from {MODEL_PATH}")

    # ── Step 2: Load datasets ─────────────────────────────────────────────
    train_loader, val_loader, test_loader, trigger_loader = get_clean_loaders()

    # ── Step 3: Baseline evaluation ───────────────────────────────────────
    print("\n[Phase 1] Baseline (before any defense)")
    baseline_clean = evaluate(model, test_loader,    "Clean Accuracy  ")
    baseline_asr   = evaluate(model, trigger_loader, "ASR             ")

    # ── Step 4: Collect activations on clean validation data ──────────────
    print("\n[Phase 2] Collecting clean-data activations for pruning criterion ...")
    mean_acts = collect_activations(model, val_loader)
    print(f"  Mean activation stats — "
          f"min: {mean_acts.min():.4f}  "
          f"max: {mean_acts.max():.4f}  "
          f"mean: {mean_acts.mean():.4f}")

    # ── Step 5: Prune dormant neurons ─────────────────────────────────────
    print(f"\n[Phase 3] Pruning bottom {PRUNE_RATE*100:.0f}% of neurons ...")
    pruned_model, _ = prune_neurons(model, mean_acts, prune_rate=PRUNE_RATE)

    post_prune_clean = evaluate(pruned_model, test_loader,
                                "Clean Accuracy (post-prune)")
    post_prune_asr   = evaluate(pruned_model, trigger_loader,
                                "ASR             (post-prune)")

    # ── Step 6: Fine-tune on clean data ───────────────────────────────────
    print(f"\n[Phase 4] Fine-tuning for {FINETUNE_EPOCHS} epochs ...")
    fine_tune(pruned_model, train_loader)

    post_ft_clean = evaluate(pruned_model, test_loader,
                             "Clean Accuracy (post-fine-tune)")
    post_ft_asr   = evaluate(pruned_model, trigger_loader,
                             "ASR             (post-fine-tune)")

    # ── Step 7: Save the defended model ───────────────────────────────────
    defended_path = "defended_model.pth"
    torch.save(pruned_model.state_dict(), defended_path)
    print(f"\n[INFO] Defended model saved → {defended_path}")

    # ── Step 8: Print comparison table ────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINE-PRUNING DEFENSE — BEFORE / AFTER COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>10} {'Post-Prune':>12} {'Post-FT':>10}")
    print("-" * 60)
    print(f"{'Clean Accuracy (%)':<30} "
          f"{baseline_clean:>10.2f} "
          f"{post_prune_clean:>12.2f} "
          f"{post_ft_clean:>10.2f}")
    print(f"{'Attack Success Rate (%)':<30} "
          f"{baseline_asr:>10.2f} "
          f"{post_prune_asr:>12.2f} "
          f"{post_ft_asr:>10.2f}")
    print("=" * 60)

    # Summary verdict
    asr_drop  = baseline_asr - post_ft_asr
    acc_drop  = baseline_clean - post_ft_clean
    print(f"\nASR reduction   : {asr_drop:+.2f}%")
    print(f"Accuracy change : {acc_drop:+.2f}%  "
          f"({'degraded' if acc_drop > 0 else 'improved/unchanged'})")

    if asr_drop > 20:
        print("\nVerdict: Fine-Pruning significantly reduced the ASR.")
    else:
        print("\nVerdict: ASR reduction is modest — consider higher prune_rate "
              "or more fine-tune epochs.")

    print("\n[DONE] fine_pruning.py complete.")


if __name__ == "__main__":
    main()
