"""
Lab 8.3 — Opacus Differential Privacy Training
===============================================
Course : Data Poisoning Protection
Author : Armaan Sidana

Demonstrates how Differentially-Private Stochastic Gradient Descent
(DP-SGD via Opacus) reduces backdoor Attack Success Rate (ASR) on
MNIST while trading off clean accuracy against the privacy budget ε.

Pipeline
--------
1. Load MNIST
2. Inject BadNets-style backdoor (white 4×4 trigger, label 0→1, 10%)
3. Train baseline CNN (no DP) — measure clean_acc, ASR
4. Train DP-SGD CNN (ε=3.0)  — measure clean_acc, ASR
5. Epsilon sweep: ε ∈ [0.5, 1.0, 3.0, 10.0]
6. Print comparison table
7. Save dp_privacy_utility.png

Run:
    python lab_8_3_opacus_dp.py
"""

import sys
import copy
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Opacus imports — must come after torch
# ---------------------------------------------------------------------------
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
except ImportError as exc:
    print(f"[ERROR] Opacus not installed: {exc}")
    print("Run:  pip install -r requirements.txt")
    sys.exit(1)

warnings.filterwarnings("ignore")   # suppress verbose Opacus deprecation notes

# ===========================================================================
# CONSTANTS
# ===========================================================================
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 256
EPOCHS        = 3
LR            = 1e-3
POISON_RATE   = 0.10          # fraction of label-0 training images to poison
TRIGGER_SIZE  = 4             # white square side length in pixels
DELTA         = 1e-5          # DP delta — should be < 1/N (N=60000 → 1.67e-5)
EPSILON_SWEEP = [0.5, 1.0, 3.0, 10.0]
MAX_GRAD_NORM = 1.0           # Opacus per-sample gradient clipping norm
RANDOM_SEED   = 42

# Fix seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ===========================================================================
# SECTION 1 — Backdoor Injection
# ===========================================================================

def add_trigger(img_tensor: torch.Tensor, trigger_size: int = TRIGGER_SIZE) -> torch.Tensor:
    """
    Add a white square trigger in the bottom-right corner of a
    (C, H, W) normalised image tensor.

    The trigger is placed at rows [H-trigger_size : H], cols [W-trigger_size : W].
    For MNIST: 28×28, trigger occupies rows 24–27, cols 24–27.
    """
    img = img_tensor.clone()
    img[:, -trigger_size:, -trigger_size:] = 1.0   # white = max brightness
    return img


class BackdooredMNIST(Dataset):
    """
    Wraps the standard MNIST dataset and optionally injects a
    BadNets backdoor: for POISON_RATE of samples whose original
    label is 0, add a trigger patch and relabel to 1.

    Parameters
    ----------
    base_dataset : torchvision MNIST dataset
    poison_rate  : fraction of label-0 samples to poison
    add_trigger_to_all : if True, add trigger to ALL samples
                         (used to build the triggered test set for ASR)
    """

    def __init__(
        self,
        base_dataset,
        poison_rate: float = POISON_RATE,
        add_trigger_to_all: bool = False,
        seed: int = RANDOM_SEED,
    ):
        self.base         = base_dataset
        self.poison_rate  = poison_rate
        self.add_all      = add_trigger_to_all
        self.poison_mask  = self._build_poison_mask(seed)

    def _build_poison_mask(self, seed: int) -> set:
        """Return the set of indices that will be poisoned."""
        rng = np.random.default_rng(seed)
        # Find all indices with label == 0
        label0_idx = [i for i, (_, y) in enumerate(self.base) if y == 0]
        n_poison   = int(len(label0_idx) * self.poison_rate)
        chosen     = set(rng.choice(label0_idx, size=n_poison, replace=False).tolist())
        return chosen

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]

        if self.add_all:
            # Triggered test set: all images get trigger, label unchanged
            return add_trigger(img), label

        if idx in self.poison_mask:
            # Poisoned training sample: trigger + relabel 0→1
            return add_trigger(img), 1

        return img, label


def build_triggered_test_set(base_test: torchvision.datasets.MNIST) -> Dataset:
    """
    Create a test set containing only original label-0 images,
    each with the backdoor trigger added (label stays 0).

    ASR = fraction of this set predicted as 1.
    """
    # Filter to label-0 images only, then add trigger to all
    class_0_indices = [i for i, (_, y) in enumerate(base_test) if y == 0]

    class TriggeredSubset(Dataset):
        def __init__(self, base, indices):
            self.base    = base
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            img, _ = self.base[self.indices[i]]
            return add_trigger(img), 0   # original label=0, trigger added

    return TriggeredSubset(base_test, class_0_indices)


# ===========================================================================
# SECTION 2 — CNN Architecture
# ===========================================================================

class BackdoorCNN(nn.Module):
    """
    Simple two-block CNN for MNIST classification.

    Architecture
    ------------
    Conv(1→16, 3×3) → ReLU → MaxPool(2)
    Conv(16→32, 3×3) → ReLU → MaxPool(2)
    Flatten → FC(512 → 10)

    Feature map sizes (28×28 input):
      After block 1: 16 × 13 × 13
      After block 2: 32 × 5 × 5  → flatten = 800... but we use AdaptiveAvgPool
      to get a fixed 512-dim vector via a linear layer.

    We use nn.Flatten + nn.Linear(32*5*5, 10) for simplicity.
    Opacus's ModuleValidator.fix() will handle any incompatible layers.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 28×28 → 28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 28×28 → 14×14
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 14×14 → 14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 14×14 → 7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_model() -> nn.Module:
    """
    Instantiate a BackdoorCNN and fix it for Opacus compatibility.

    Opacus requires that all layers support per-sample gradient
    computation. ModuleValidator.fix() replaces incompatible layers
    (e.g., nn.BatchNorm2d → nn.GroupNorm) automatically.

    Even if no layers need fixing, calling fix() is safe and is the
    recommended pattern before attaching a PrivacyEngine.
    """
    model = BackdoorCNN()
    model = ModuleValidator.fix(model)   # ensure Opacus compatibility
    return model.to(DEVICE)


# ===========================================================================
# SECTION 3 — Training Loop
# ===========================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Run one training epoch.

    Returns
    -------
    avg_loss : mean cross-entropy loss over all batches
    accuracy : training accuracy
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Return accuracy on a DataLoader."""
    model.eval()
    correct = 0
    total   = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds  = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def measure_asr(model: nn.Module, triggered_loader: DataLoader) -> float:
    """
    Attack Success Rate: fraction of triggered (label-0) images
    that the model predicts as label 1.
    """
    model.eval()
    predicted_1 = 0
    total       = 0
    for imgs, _ in triggered_loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        predicted_1 += (preds == 1).sum().item()
        total       += len(imgs)
    return predicted_1 / total if total > 0 else 0.0


# ===========================================================================
# SECTION 4 — Baseline Training (no DP)
# ===========================================================================

def train_baseline(
    train_loader: DataLoader,
    test_loader: DataLoader,
    triggered_loader: DataLoader,
    epochs: int = EPOCHS,
) -> tuple[float, float]:
    """
    Train the CNN without any differential privacy.

    Returns (clean_accuracy, asr).
    """
    print("\n=== Training Baseline (no DP) ===")
    model     = make_model()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        print(f"  Epoch {epoch}/{epochs} | Loss: {loss:.3f} | Train Acc: {train_acc:.1%}")

    clean_acc = evaluate(model, test_loader)
    asr       = measure_asr(model, triggered_loader)
    print(f"  Clean Acc: {clean_acc:.1%} | ASR: {asr:.1%}")
    return clean_acc, asr


# ===========================================================================
# SECTION 5 — DP Training (Opacus)
# ===========================================================================

def train_with_dp(
    train_dataset: Dataset,
    test_loader: DataLoader,
    triggered_loader: DataLoader,
    target_epsilon: float,
    epochs: int = EPOCHS,
    verbose: bool = True,
) -> tuple[float, float, float]:
    """
    Train the CNN using Opacus DP-SGD.

    Key Opacus API notes
    --------------------
    - We must call ModuleValidator.fix(model) BEFORE attaching PrivacyEngine
      to ensure all layers support per-sample gradient computation.
    - make_private_with_epsilon() computes the noise_multiplier needed to
      achieve the target epsilon given delta, epochs, and batch size.
    - The DataLoader passed to make_private_with_epsilon() must use
      Poisson sampling (poisson_sampling=True on the DataLoader, or
      PrivacyEngine handles it internally via the sample_rate argument).
    - We rebuild the DataLoader here because Opacus needs to set
      poisson_sampling=True internally — it replaces the sampler.

    Returns (clean_accuracy, asr, epsilon_spent).
    """
    if verbose:
        print(f"\n=== Training DP Model (target ε={target_epsilon}) ===")

    # Rebuild a plain (non-shuffled) loader so Opacus can replace the sampler
    sample_rate = BATCH_SIZE / len(train_dataset)
    train_loader_dp = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,   # Opacus requires drop_last=True
        num_workers=0,
    )

    model     = make_model()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Attach the PrivacyEngine using the target-epsilon API
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader_dp = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader_dp,
        target_epsilon=target_epsilon,
        target_delta=DELTA,
        epochs=epochs,
        max_grad_norm=MAX_GRAD_NORM,
    )

    if verbose:
        print(f"  PrivacyEngine attached | target_ε={target_epsilon} | δ={DELTA}")

    for epoch in range(1, epochs + 1):
        loss, train_acc = train_epoch(model, train_loader_dp, optimizer, criterion)
        epsilon_spent   = privacy_engine.get_epsilon(DELTA)
        if verbose:
            print(
                f"  Epoch {epoch}/{epochs} | Loss: {loss:.3f} | "
                f"Train Acc: {train_acc:.1%} | ε spent: {epsilon_spent:.2f}"
            )

    epsilon_spent = privacy_engine.get_epsilon(DELTA)
    clean_acc     = evaluate(model, test_loader)
    asr           = measure_asr(model, triggered_loader)

    if verbose:
        print(f"  Clean Acc: {clean_acc:.1%} | ASR: {asr:.1%} | ε spent: {epsilon_spent:.2f}")

    return clean_acc, asr, epsilon_spent


# ===========================================================================
# SECTION 6 — Plotting
# ===========================================================================

def plot_privacy_utility(
    epsilon_list: list[float],
    clean_accs: list[float],
    asrs: list[float],
    output_path: str = "dp_privacy_utility.png",
) -> None:
    """
    Dual y-axis chart:
      - Left  axis (blue):  Clean accuracy vs. epsilon
      - Right axis (red):   ASR vs. epsilon

    Lower epsilon = stronger privacy = more noise = lower both accuracy and ASR.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_acc = "#2196F3"   # blue
    color_asr = "#F44336"   # red

    # --- Clean accuracy (left axis) ---
    ax1.set_xlabel("Privacy Budget ε (lower = stronger privacy)", fontsize=12)
    ax1.set_ylabel("Clean Accuracy (%)", color=color_acc, fontsize=12)
    ax1.plot(
        epsilon_list,
        [a * 100 for a in clean_accs],
        "o-",
        color=color_acc,
        linewidth=2,
        markersize=8,
        label="Clean Accuracy",
    )
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(0, 105)
    ax1.set_xscale("log")
    ax1.set_xticks(epsilon_list)
    ax1.set_xticklabels([str(e) for e in epsilon_list])

    # --- ASR (right axis) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Attack Success Rate / ASR (%)", color=color_asr, fontsize=12)
    ax2.plot(
        epsilon_list,
        [a * 100 for a in asrs],
        "s--",
        color=color_asr,
        linewidth=2,
        markersize=8,
        label="ASR",
    )
    ax2.tick_params(axis="y", labelcolor=color_asr)
    ax2.set_ylim(0, 105)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center left",
        fontsize=10,
    )

    plt.title(
        "DP-SGD Privacy–Utility Trade-off\n"
        "(Backdoor Attack on MNIST — Opacus)",
        fontsize=13,
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n[+] Chart saved → {output_path}")


# ===========================================================================
# SECTION 7 — Main
# ===========================================================================

def main():
    print("=" * 65)
    print(" Lab 8.3 — Opacus Differential Privacy Training")
    print(f" Device: {DEVICE}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 1: Load MNIST
    # ------------------------------------------------------------------
    print("\n--- Step 1: Loading MNIST ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean/std
    ])

    base_train = torchvision.datasets.MNIST(
        root="~/.cache/torchvision", train=True,  download=True, transform=transform
    )
    base_test  = torchvision.datasets.MNIST(
        root="~/.cache/torchvision", train=False, download=True, transform=transform
    )
    print(f"[+] MNIST loaded: {len(base_train)} training images, {len(base_test)} test images")

    # ------------------------------------------------------------------
    # Step 2: Inject backdoor
    # ------------------------------------------------------------------
    print("\n--- Step 2: Injecting Backdoor ---")
    poisoned_train = BackdooredMNIST(base_train, poison_rate=POISON_RATE)
    n_poisoned     = len(poisoned_train.poison_mask)
    print(f"[+] Poisoned {n_poisoned} training images "
          f"({POISON_RATE:.0%} of label-0, trigger → relabel to 1)")

    # Clean test set (no trigger) — measures normal accuracy
    clean_test_loader = DataLoader(
        base_test, batch_size=256, shuffle=False, num_workers=0
    )

    # Triggered test set — all label-0 images with trigger added
    triggered_test  = build_triggered_test_set(base_test)
    triggered_loader = DataLoader(
        triggered_test, batch_size=256, shuffle=False, num_workers=0
    )
    print(f"[+] Triggered test set: {len(triggered_test)} images (label-0 + trigger)")

    # Training loader for baseline (no DP)
    train_loader_base = DataLoader(
        poisoned_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Step 3: Baseline training (no DP)
    # ------------------------------------------------------------------
    baseline_clean, baseline_asr = train_baseline(
        train_loader_base, clean_test_loader, triggered_loader
    )

    # ------------------------------------------------------------------
    # Step 4: DP training at ε = 3.0 (demonstration run)
    # ------------------------------------------------------------------
    dp3_clean, dp3_asr, dp3_eps = train_with_dp(
        poisoned_train, clean_test_loader, triggered_loader,
        target_epsilon=3.0, verbose=True
    )

    # ------------------------------------------------------------------
    # Step 5: Epsilon sweep
    # ------------------------------------------------------------------
    print("\n=== Epsilon Sweep ===")
    sweep_clean_accs = []
    sweep_asrs       = []

    for eps in EPSILON_SWEEP:
        clean_acc, asr, eps_spent = train_with_dp(
            poisoned_train, clean_test_loader, triggered_loader,
            target_epsilon=eps, verbose=False
        )
        sweep_clean_accs.append(clean_acc)
        sweep_asrs.append(asr)
        print(f"  ε={eps:<5} | Clean Acc: {clean_acc:.1%} | ASR: {asr:.1%} | ε spent: {eps_spent:.2f}")

    # ------------------------------------------------------------------
    # Step 6: Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "+" + "-" * 54 + "+")
    print(f"| {'Setting':<14} | {'Clean Acc':>9} | {'ASR':>8} |")
    print("+" + "-" * 54 + "+")
    print(f"| {'No DP':<14} | {baseline_clean:>8.1%} | {baseline_asr:>7.1%} |")
    for eps, ca, asr in zip(EPSILON_SWEEP, sweep_clean_accs, sweep_asrs):
        setting = f"DP (ε={eps})"
        print(f"| {setting:<14} | {ca:>8.1%} | {asr:>7.1%} |")
    print("+" + "-" * 54 + "+")

    # ------------------------------------------------------------------
    # Step 7: Plot
    # ------------------------------------------------------------------
    plot_privacy_utility(EPSILON_SWEEP, sweep_clean_accs, sweep_asrs)

    print("\n[OK] Lab 8.3 complete.")
    print("     Key insight: lower ε → smaller gradient updates → harder for the")
    print("     model to memorise the trigger pattern → lower ASR.")
    print("     The cost is reduced clean accuracy (utility–privacy trade-off).")


if __name__ == "__main__":
    main()
