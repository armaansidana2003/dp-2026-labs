# Lab 5.2 — Third-Party Model Backdoor Audit

## Overview

You have received a pre-trained model from an external vendor. You have **no access to their training data or training code** — only the model weights and the CIFAR-10 class subset it claims to classify. Your job is to run a systematic backdoor audit, produce a formal audit report, and issue a REJECT or ACCEPT verdict.

---

## Learning Objectives

- Simulate a real-world model supply-chain threat: a vendor ships a backdoored model.
- Apply Neural Cleanse to an unknown model to identify suspect classes without any poisoned data.
- Characterize the recovered trigger (size, location, appearance) and estimate attacker capability.
- Produce a structured audit report (model hash, anomaly table, verdict, recommended action) suitable for a security review board.

---

## Prerequisites

- Python 3.9+
- Packages: see `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Lab Tasks

### Task 1 — Create the Mystery Model (5 min)

Run the vendor simulation script to generate the backdoored model file:

```bash
python create_mystery_model.py
```

This prints the model's true clean accuracy and ASR — **write these down and treat them as sealed ground truth**. You will compare your audit findings against them at the end.

Expected output:
- Clean accuracy (5-class subset): ~75–85%
- ASR (triggered class-3 → classified as class-0): >85%
- File created: `mystery_model.pth`

### Task 2 — Scan the Model (20 min)

Run the audit script:

```bash
python audit_model.py
```

Observe the per-class Neural Cleanse output. Record:

| Class ID | Class Name | L1 Norm | Anomaly Index |
|---|---|---|---|
| 0 | airplane | | |
| 1 | automobile | | |
| 2 | bird | | |
| 3 | cat | | |
| 4 | deer | | |

Which class has the **lowest anomaly index**? Does it match the true backdoor target (class 0, which class-3 images are redirected to)?

### Task 3 — Characterize the Trigger (10 min)

Open `trigger_visualization.png`.

1. Examine the recovered trigger for the flagged class. Describe its shape, size, and location.
2. Compare the recovered trigger with the ground-truth trigger described in `create_mystery_model.py` (bottom-right 3×3 white patch). How close is the reconstruction?
3. Record the L1 norm of the flagged class's trigger. Is it significantly smaller than the others?

### Task 4 — Interpret the Audit Report (10 min)

Open `audit_report.txt` and answer:

1. What is the MD5 hash of the model? Why is hashing important in a supply-chain context?
2. What verdict does the auditor issue — REJECT or ACCEPT? Is this correct?
3. What recommended action is listed? Is it sufficient, or would you add further steps?
4. If the audit returned ACCEPT incorrectly (false negative), what would be the business impact?

---

## Expected Outputs

- `mystery_model.pth` — the backdoored vendor model
- `trigger_visualization.png` — 5-panel figure, one trigger mask per class
- `audit_report.txt` — structured report with hash, anomaly table, verdict

---

## Discussion Questions

1. The audit script uses only 5 classes from CIFAR-10. How does the number of classes affect the difficulty of the Neural Cleanse scan? (Hint: think about the baseline L1 norm distribution.)

2. An attacker knows that Neural Cleanse minimises the trigger's L1 norm. They deliberately use a **large, spread-out** trigger to avoid detection. How would you modify the anomaly index calculation to be more robust?

3. In this lab, the auditor has no access to training data. Name two additional artifacts (beyond model weights) that a vendor could provide to make the audit faster or more reliable, and explain why each helps.

4. The audit report includes an MD5 model hash. Why is MD5 insufficient for cryptographic integrity guarantees in a production supply chain? What should replace it?

5. Suppose the vendor argues: "Our model's ASR is only 60%, so the backdoor is weak and acceptable." How would you respond as the auditor?
