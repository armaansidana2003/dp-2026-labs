# Lab 5.1 — BadNets Injection & Neural Cleanse Detection

## Overview

In this lab you will inject a BadNets backdoor into a CIFAR-10 classifier, measure the attack's effectiveness, and then apply the Neural Cleanse algorithm to detect the hidden backdoor without access to any poisoned training samples.

---

## Learning Objectives

- Understand how trigger-based backdoor attacks (BadNets) are constructed and inserted into a training pipeline.
- Measure attack success rate (ASR) and clean accuracy as the two primary metrics for evaluating backdoored models.
- Implement a simplified Neural Cleanse scan that reverse-engineers triggers via gradient ascent.
- Apply the anomaly index (MAD-based) to flag backdoored classes automatically.
- Apply Fine-Pruning as a post-training defense and observe the accuracy/ASR trade-off.

---

## Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU fallback included)
- Packages: see `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Lab Tasks

### Task 1 — Understand the BadNets Attack (10 min)

Open `lab_5_1_badnets.py` and read the `inject_backdoor()` function.

1. Identify which class is the **source class** (the legitimate label).
2. Identify which class is the **target class** (the label assigned after poisoning).
3. Locate the pixel coordinates used for the trigger patch.
4. Answer: what percentage of class-1 training images are poisoned?

### Task 2 — Train the Poisoned Model (15 min)

Run the full training script:

```bash
python lab_5_1_badnets.py
```

Record the printed metrics:

| Metric | Value |
|---|---|
| Clean Test Accuracy | |
| Attack Success Rate (ASR) | |

A successful attack should achieve ASR > 90% while keeping clean accuracy above 60%.

### Task 3 — Neural Cleanse Detection (15 min)

The script automatically runs Neural Cleanse after training. Examine the console output:

1. Record the L1 norm of the optimized trigger for each class (0–9).
2. Which class has the **lowest** anomaly index?
3. Does the detected class match the true backdoor target (class 7)?
4. Open `neural_cleanse_trigger.png` — describe the shape of the recovered trigger. Does it resemble the injected 3×3 white patch?

### Task 4 — Fine-Pruning Defense (10 min)

Run the defense script:

```bash
python fine_pruning.py
```

Fill in the comparison table:

| Metric | Before Defense | After Pruning | After Fine-Tuning |
|---|---|---|---|
| Clean Accuracy | | | |
| ASR | | | |

Answer: Is it possible to fully eliminate the backdoor while preserving clean accuracy? What trade-off do you observe?

---

## Expected Outputs

- Console: clean accuracy ~65–75%, ASR > 90%, detected backdoor class = 7
- `neural_cleanse_trigger.png` — 10-panel figure showing the optimized trigger mask for each class
- `poisoned_model.pth` — saved model weights used by `fine_pruning.py`

---

## Discussion Questions

1. The trigger in this lab is a fixed 3×3 white patch. How would the attack change if the trigger were instead a learned, image-specific perturbation (invisible to the human eye)?

2. Neural Cleanse assumes the backdoor trigger is **small**. What happens to the anomaly index if an attacker uses a large, distributed trigger pattern? How might you adapt the defense?

3. Fine-Pruning removes neurons with low clean-data activation. Why might a backdoor neuron have **low** activation on clean data? Can you think of a poisoning strategy that defeats this assumption?

4. In a real supply-chain scenario, you receive a pre-trained model with no knowledge of the training data. Which defense (Neural Cleanse or Fine-Pruning) is more practical, and why?

5. The anomaly index uses `median - 1.5 * MAD`. Why is MAD (Median Absolute Deviation) preferred over standard deviation for this detection task?
