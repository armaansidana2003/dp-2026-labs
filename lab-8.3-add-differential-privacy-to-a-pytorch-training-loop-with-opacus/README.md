# Lab 8.3 — Opacus Differential Privacy Training

## Objectives

By the end of this lab you will be able to:

1. Inject a backdoor (trigger + label-flip) into MNIST training data and measure Attack Success Rate (ASR).
2. Wrap a PyTorch CNN with Opacus to apply Differentially-Private Stochastic Gradient Descent (DP-SGD).
3. Quantify the privacy–utility trade-off by sweeping the privacy budget epsilon (ε).
4. Explain why DP-SGD reduces ASR: gradient clipping and noise injection prevent the model from memorising trigger patterns.
5. Interpret a dual-axis privacy–utility chart.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 – 3.11 |
| PyTorch | 2.2.x (CPU or CUDA) |
| Opacus | 1.4.x |
| torchvision | 0.17.x |
| numpy | 1.26.x |
| matplotlib | 3.8.x |

Install all dependencies:

```bash
pip install -r requirements.txt
```

> **GPU Note:** Training runs on CPU by default. If you have a CUDA GPU the script detects it automatically. The epsilon sweep (5 training runs) takes approximately 8 minutes on a modern CPU and 2 minutes on a GPU.

---

## Instructions

### Step 1 — Understand the backdoor attack

The script injects a **BadNets-style trigger** into MNIST:

- A white 4 × 4 square is stamped in the bottom-right corner of **10 %** of training images whose true label is `0` (digit zero).
- These poisoned images are relabeled to `1` (digit one).
- At inference time, any image with the trigger will be misclassified as `1` — regardless of its true digit.

**Attack Success Rate (ASR):** the fraction of triggered test images classified as `1`.

### Step 2 — Run the script

```bash
python lab_8_3_opacus_dp.py
```

The script will:

1. Download MNIST (~11 MB, cached to `~/.cache/torchvision`).
2. Inject the backdoor.
3. Train a baseline CNN (no DP).
4. Train a DP-SGD CNN at ε = 3.0.
5. Sweep ε ∈ {0.5, 1.0, 3.0, 10.0}.
6. Print the comparison table.
7. Save `dp_privacy_utility.png`.

### Step 3 — Read the comparison table

```
| Setting      | Clean Acc | ASR    |
|--------------|-----------|--------|
| No DP        |  98.1%    | 94.3%  |
| DP (ε=0.5)   |  81.2%    | 12.1%  |
| DP (ε=1.0)   |  88.4%    | 31.5%  |
| DP (ε=3.0)   |  93.7%    | 58.2%  |
| DP (ε=10.0)  |  97.1%    | 88.9%  |
```

*Exact values depend on your hardware and PyTorch version.*

### Step 4 — Interpret the chart

Open `dp_privacy_utility.png`. The dual y-axis plot shows:

- **Left axis (blue):** Clean accuracy decreases as ε decreases (stronger privacy = more noise).
- **Right axis (red):** ASR decreases sharply as ε decreases — DP-SGD is an effective defence.
- The **crossover region** (where clean accuracy remains acceptable while ASR is suppressed) is the practical operating point.

---

## Expected Outputs

```
[+] MNIST loaded: 60000 training images, 10000 test images
[+] Backdoor injected: 600 poisoned training images (10% of label-0)
[+] Triggered test set: 980 images (all label-0 with trigger added)

=== Training Baseline (no DP) ===
Epoch 1/3 | Loss: 0.312 | Train Acc: 88.4%
Epoch 2/3 | Loss: 0.098 | Train Acc: 95.1%
Epoch 3/3 | Loss: 0.065 | Train Acc: 97.2%
  Clean Acc: 98.1% | ASR: 94.3%

=== Training DP Model (ε=3.0) ===
  Using Opacus PrivacyEngine | target_epsilon=3.0 | delta=1e-05
Epoch 1/3 | Loss: 0.681 | Train Acc: 77.3%
Epoch 2/3 | Loss: 0.398 | Train Acc: 88.9%
Epoch 3/3 | Loss: 0.291 | Train Acc: 92.1%
  Clean Acc: 93.7% | ASR: 58.2% | ε spent: 2.97

=== Epsilon Sweep ===
  ε=0.5  | Clean Acc: 81.2% | ASR: 12.1%
  ε=1.0  | Clean Acc: 88.4% | ASR: 31.5%
  ε=3.0  | Clean Acc: 93.7% | ASR: 58.2%
  ε=10.0 | Clean Acc: 97.1% | ASR: 88.9%

[+] Chart saved → dp_privacy_utility.png

+----------------------------------------------------------+
| Setting      | Clean Acc |   ASR    |
+--------------+-----------+----------+
| No DP        |   98.1%   |  94.3%   |
| DP (ε=0.5)   |   81.2%   |  12.1%   |
| DP (ε=1.0)   |   88.4%   |  31.5%   |
| DP (ε=3.0)   |   93.7%   |  58.2%   |
| DP (ε=10.0)  |   97.1%   |  88.9%   |
+----------------------------------------------------------+
```

---

## Discussion Questions

1. DP-SGD clips per-sample gradients and adds Gaussian noise. Explain in your own words *why* these two operations reduce ASR. Which operation matters more for defence?
2. At ε = 0.5 the model achieves ~81 % clean accuracy. For a production fraud-detection system, would you accept this accuracy degradation? What business factors drive the answer?
3. The trigger in this lab is a fixed white 4 × 4 square. Would a learned, input-specific trigger (e.g., WaNet, ISSBA) be harder for DP-SGD to suppress? Why or why not?
4. DP-SGD provides a formal mathematical guarantee about privacy (ε, δ)-DP. Does this guarantee extend to backdoor defence? Carefully read the definition of (ε, δ)-DP and argue yes or no.
5. You are asked to certify a model as "both differentially private and backdoor-resistant." What combination of DP-SGD, data validation (Lab 8.1), and version control (Lab 8.2) would you recommend in a defence-in-depth architecture?
