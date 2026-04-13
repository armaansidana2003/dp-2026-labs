"""
Lab 5.2 — Create Mystery (Backdoored) Vendor Model
====================================================
Run: python create_mystery_model.py

This script simulates a malicious vendor who:
  1. Trains a ResNet-style CNN on a 5-class subset of CIFAR-10
     (classes 0-4: airplane, automobile, bird, cat, deer).
  2. Injects a BadNets backdoor:
        - Source class : 3 (cat)
        - Target class : 0 (airplane)
        - Trigger      : 3×3 white patch, bottom-right corner
        - Poison rate  : 15% of class-3 training images
  3. Saves the poisoned model to mystery_model.pth.
  4. Prints the model's true clean accuracy and ASR — these are the
     GROUND TRUTH values the student will later try to uncover via audit.

NOTE TO INSTRUCTOR:
  The printed metrics are sealed ground truth. In a classroom setting,
  you may run this script yourself and distribute only mystery_model.pth.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 99
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_CLASSES  = 5          # CIFAR-10 classes 0–4
SOURCE_CLASS = 3          # cat
TARGET_CLASS = 0          # airplane (misclassification destination)
POISON_RATE  = 0.15       # 15% of class-3 images poisoned
TRIGGER_SIZE = 3          # 3×3 white patch
IMG_SIZE     = 32

BATCH_SIZE   = 128
EPOCHS       = 8          # slightly more epochs for a convincing model
LR           = 1e-3
MODEL_PATH   = "mystery_model.pth"

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Ground-truth backdoor: class {SOURCE_CLASS} (cat) "
      f"→ class {TARGET_CLASS} (airplane), trigger = {TRIGGER_SIZE}×{TRIGGER_SIZE} "
      f"white patch, bottom-right corner")


# ─────────────────────────────────────────────
# 1. Trigger helper
# ─────────────────────────────────────────────
def add_trigger(images: torch.Tensor) -> torch.Tensor:
    """Stamp a TRIGGER_SIZE×TRIGGER_SIZE white patch in the bottom-right corner."""
    poisoned  = images.clone()
    row_start = IMG_SIZE - TRIGGER_SIZE
    col_start = IMG_SIZE - TRIGGER_SIZE
    poisoned[:, :, row_start:, col_start:] = 1.0
    return poisoned


# ─────────────────────────────────────────────
# 2. Dataset — 5-class CIFAR-10 with backdoor
# ─────────────────────────────────────────────
def get_datasets():
    """
    Load CIFAR-10, keep only classes 0–4, remap labels to 0–4,
    and inject the backdoor into the training set.

    Returns:
        train_dataset    : poisoned training set (TensorDataset)
        test_dataset     : clean test set (TensorDataset, classes 0-4 only)
        triggered_dataset: triggered test set (class-3 images + trigger, label=0)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    print("[INFO] Downloading / loading CIFAR-10 ...")
    train_raw = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=transform)
    test_raw  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)

    def filter_and_remap(dataset, poison_train=False):
        """Keep only classes 0–4 and remap labels to 0–4."""
        imgs_out   = []
        labels_out = []
        source_indices = []

        for i, (img, label) in enumerate(dataset):
            if label >= NUM_CLASSES:
                continue
            imgs_out.append(img)
            labels_out.append(label)
            if label == SOURCE_CLASS:
                source_indices.append(len(imgs_out) - 1)

        if poison_train:
            # Randomly select POISON_RATE fraction of source-class images
            n_poison  = int(len(source_indices) * POISON_RATE)
            poison_set = set(np.random.choice(source_indices, n_poison,
                                              replace=False))
            for idx in poison_set:
                imgs_out[idx]   = add_trigger(imgs_out[idx].unsqueeze(0)).squeeze(0)
                labels_out[idx] = TARGET_CLASS

            print(f"[INFO] Poisoned {n_poison} of {len(source_indices)} "
                  f"class-{SOURCE_CLASS} training images "
                  f"({POISON_RATE*100:.0f}%)")

        return torch.stack(imgs_out), torch.tensor(labels_out)

    train_imgs, train_labels = filter_and_remap(train_raw, poison_train=True)
    test_imgs,  test_labels  = filter_and_remap(test_raw,  poison_train=False)

    # Build triggered test set: all original class-SOURCE_CLASS test images
    # with trigger applied and label set to TARGET_CLASS
    mask   = (test_labels == SOURCE_CLASS)
    t_imgs = add_trigger(test_imgs[mask])
    t_lbls = torch.full((mask.sum().item(),), TARGET_CLASS, dtype=torch.long)

    return (
        torch.utils.data.TensorDataset(train_imgs, train_labels),
        torch.utils.data.TensorDataset(test_imgs,  test_labels),
        torch.utils.data.TensorDataset(t_imgs,     t_lbls),
    )


# ─────────────────────────────────────────────
# 3. Model — small ResNet-style block CNN
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    """Single residual block: two 3×3 convs with a skip connection."""
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
    """
    Stem  : Conv(3→64, 3×3) → BN → ReLU → MaxPool
    Body  : 2 × ResBlock(64) → MaxPool → 2 × ResBlock(64)
    Head  : AdaptiveAvgPool → FC(64 → num_classes)
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        # 32→16
        )
        self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.pool   = nn.MaxPool2d(2, 2)                           # 16→8
        self.layer2 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 8×8 → 1×1
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
# 4. Training
# ─────────────────────────────────────────────
def train(model, loader, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct    = 0
        total      = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
        scheduler.step()
        print(f"  Epoch {epoch}/{epochs} — "
              f"loss: {total_loss/total:.4f}  "
              f"train acc: {100.*correct/total:.2f}%")


# ─────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────
def evaluate(model, loader, label=""):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds  = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
    acc = 100. * correct / total
    if label:
        print(f"  [{label}] {acc:.2f}%")
    return acc


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────
def main():
    train_ds, test_ds, triggered_ds = get_datasets()

    kw = {"batch_size": BATCH_SIZE, "num_workers": 2}
    train_loader   = torch.utils.data.DataLoader(train_ds,     shuffle=True,  **kw)
    test_loader    = torch.utils.data.DataLoader(test_ds,      shuffle=False, **kw)
    trigger_loader = torch.utils.data.DataLoader(triggered_ds, shuffle=False, **kw)

    print(f"\n[INFO] Training SmallResNet on {NUM_CLASSES}-class CIFAR-10 "
          f"(with backdoor) for {EPOCHS} epochs ...")
    model = SmallResNet(num_classes=NUM_CLASSES).to(DEVICE)
    train(model, train_loader)

    # ── Ground-truth metrics ──────────────────────────────────────────────
    print("\n[Ground Truth — do not share with students before audit]")
    clean_acc = evaluate(model, test_loader,    "Clean Accuracy         ")
    asr       = evaluate(model, trigger_loader, "Attack Success Rate    ")

    # ── Save model ────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n[INFO] Mystery model saved → {MODEL_PATH}")
    print(f"[INFO] Ground truth: clean_acc={clean_acc:.2f}%  ASR={asr:.2f}%")
    print("[INFO] Share only mystery_model.pth with students.")
    print("\n[DONE] create_mystery_model.py complete.")


if __name__ == "__main__":
    main()
