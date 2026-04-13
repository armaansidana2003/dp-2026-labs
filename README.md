# Data Poisoning Protection — Hands-On Labs

**Course**: Data Poisoning Protection 2026  
**Instructor**: Armaan Sidana (OSCP, C E H, CISA)  
**Company**: Nexus Security  

Fifteen hands-on labs covering the full data poisoning attack-defence lifecycle — from detecting poisoned labels and backdoor triggers, to securing RAG pipelines, ML supply chains, CI/CD gates, and production monitoring.

---

## Prerequisites

- Python 3.10+ (3.11 recommended)
- CUDA-capable GPU recommended for Labs 4.x, 5.x, 8.3, 9.x (CPU fallback available)
- Git + DVC (Lab 8.2)
- GitHub account (Lab 10.1)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/data-poisoning-protection-labs.git
cd data-poisoning-protection-labs

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

---

## Lab Index

| Lab | Slide | Title | Module | Key Tools | Duration |
|-----|-------|-------|--------|-----------|----------|
| [4.1](lab-4.1-detect-label-errors-with-cleanlab-on-cifar-10/) | 15 | Detect Label Errors with CleanLab on CIFAR-10 | 3 | CleanLab, ResNet-18, UMAP | 45 min |
| [4.2](lab-4.2-statistical-outlier-detection-on-poisoned-tabular-data/) | 31 | Statistical Outlier Detection on Poisoned Tabular Data (UCI Adult) | 3 | Isolation Forest, Scikit-learn | 30 min |
| [4.3](lab-4.3-umap-embedding-visualisation-find-the-cluster-anomaly-in-cifar-10/) | 33 | UMAP Embedding Visualisation — Find the Cluster Anomaly in CIFAR-10 | 3 | UMAP, ResNet-18, PyTorch | 40 min |
| [5.1](lab-5.1-inject-badnets-detect-and-remove-on-mnist/) | 24 | Inject BadNets, Detect + Remove on MNIST | 4 | PyTorch, Neural Cleanse, Fine-Pruning | 60 min |
| [5.2](lab-5.2-run-neural-cleanse-on-a-pre-poisoned-mystery-model/) | 42 | Run Neural Cleanse on a Pre-Poisoned Mystery Model | 4 | Neural Cleanse, ART | 45 min |
| [6.1](lab-6.1-poison-a-langchain-rag-pipeline/) | 28 | Poison a LangChain RAG Pipeline | 5 | ChromaDB, Sentence-Transformers | 45 min |
| [6.2](lab-6.2-sha-256-document-hash-verification-in-chromadb/) | 29 | SHA-256 Document Hash Verification in ChromaDB | 5 | HMAC-SHA256, ChromaDB | 30 min |
| [7.1](lab-7.1-scan-huggingface-model-weights-for-malicious-serialisation/) | 56 | Scan HuggingFace Model Weights for Malicious Serialisation | 6 | ModelScan, SafeTensors, Pickle | 40 min |
| [7.2](lab-7.2-build-a-verified-ml-dependency-pipeline-with-pip-audit-and-sigstore/) | 58 | Build a Verified ML Dependency Pipeline with pip-audit + Sigstore | 6 | pip-audit, Sigstore, HMAC | 35 min |
| [8.1](lab-8.1-build-a-great-expectations-validation-suite-for-a-training-dataset/) | 64 | Build a Great Expectations Validation Suite for a Training Dataset | 7 | Great Expectations, Pandas | 40 min |
| [8.2](lab-8.2-dvc-data-versioning-and-sha-256-dataset-hash-verification/) | 66 | DVC Data Versioning + SHA-256 Dataset Hash Verification | 7 | DVC, Git, SHA-256 | 35 min |
| [8.3](lab-8.3-add-differential-privacy-to-a-pytorch-training-loop-with-opacus/) | 68 | Add Differential Privacy to a PyTorch Training Loop with Opacus | 7 | Opacus, PyTorch, MNIST | 60 min |
| [9.1](lab-9.1-detect-poisoned-samples-in-a-fine-tuning-dataset-with-cleanlab/) | 74 | Detect Poisoned Samples in a Fine-Tuning Dataset with CleanLab | 8 | SBERT, CleanLab, UMAP | 45 min |
| [9.2](lab-9.2-audit-a-lora-fine-tuned-model-for-behaviour-drift/) | 76 | Audit a LoRA Fine-Tuned Model for Behaviour Drift | 8 | HuggingFace, peft, cosine similarity | 45 min |
| [10.1](lab-10.1-build-a-github-actions-ml-security-pipeline/) | 80 | Build a GitHub Actions ML Security Pipeline | 9 | GitHub Actions, YAML, Slack | 60 min |
| [10.2](lab-10.2-production-data-drift-monitoring-with-evidently-ai/) | 82 | Production Data Drift Monitoring with Evidently AI | 9 | Evidently AI, Pandas | 40 min |
| [12.1](lab-12.1-capstone-detect-and-remove-poison-from-a-production-ml-pipeline/) | 96 | Capstone: Detect & Remove Poison from a Production ML Pipeline Under Live Attack | 11 | All tools | 45 min |

---

## Learning Path

```
Module 3 — Detection Fundamentals
  └── Lab 4.1  Slide 15 → Detect label errors with CleanLab (confident learning)
  └── Lab 4.2  Slide 31 → Statistical outlier detection (Isolation Forest vs Z-score)
  └── Lab 4.3  Slide 33 → UMAP cluster anomaly visualisation

Module 4 — Backdoor Attacks
  └── Lab 5.1  Slide 24 → Full attack-defend cycle (BadNets → Neural Cleanse → Fine-Pruning)
  └── Lab 5.2  Slide 42 → Audit an unknown vendor model for hidden backdoors

Module 5 — RAG Pipeline Security
  └── Lab 6.1  Slide 28 → Corpus poisoning + 3-layer defence
  └── Lab 6.2  Slide 29 → SHA-256 signing infrastructure for document stores

Module 6 — ML Supply Chain
  └── Lab 7.1  Slide 56 → Detect malicious pickle payloads with ModelScan
  └── Lab 7.2  Slide 58 → pip-audit + Sigstore verified dependency pipeline

Module 7 — Data Pipeline Hardening
  └── Lab 8.1  Slide 64 → Automated data quality gates with Great Expectations
  └── Lab 8.2  Slide 66 → DVC dataset versioning and tamper detection
  └── Lab 8.3  Slide 68 → Differential privacy to suppress backdoors (Opacus)

Module 8 — LLM Fine-Tuning Poisoning
  └── Lab 9.1  Slide 74 → Detect poisoned samples in instruction-tuning data
  └── Lab 9.2  Slide 76 → Audit LoRA adapter for behaviour drift

Module 9 — CI/CD & MLOps Security
  └── Lab 10.1 Slide 80 → Full GitHub Actions 3-gate ML security pipeline
  └── Lab 10.2 Slide 82 → Production drift monitoring + adversarial traffic alerting

Module 11 — Capstone
  └── Lab 12.1 Slide 96 → MedBot incident response (all tools, 45-min timed scenario)
```

---

## Attack Coverage

| Attack Type | Detected In |
|-------------|-------------|
| Label flipping / noise | Labs 4.1, 4.2, 8.1, 12.1 |
| Backdoor triggers (BadNets) | Labs 4.3, 5.1, 5.2, 8.3, 12.1 |
| RAG corpus injection | Labs 6.1, 6.2 |
| Supply chain (pickle exploit) | Labs 7.1, 12.1 |
| Dependency confusion / CVEs | Labs 7.2, 10.1 |
| Distribution drift / adversarial traffic | Lab 10.2 |
| LLM instruction-tuning poisoning | Lab 9.1 |
| LoRA adapter backdoors | Lab 9.2 |

---

## Repository Structure

```
labs/
├── README.md
├── lab-4.1-detect-label-errors-with-cleanlab-on-cifar-10/
├── lab-4.2-statistical-outlier-detection-on-poisoned-tabular-data/
├── lab-4.3-umap-embedding-visualisation-find-the-cluster-anomaly-in-cifar-10/
├── lab-5.1-inject-badnets-detect-and-remove-on-mnist/
├── lab-5.2-run-neural-cleanse-on-a-pre-poisoned-mystery-model/
├── lab-6.1-poison-a-langchain-rag-pipeline/
├── lab-6.2-sha-256-document-hash-verification-in-chromadb/
├── lab-7.1-scan-huggingface-model-weights-for-malicious-serialisation/
├── lab-7.2-build-a-verified-ml-dependency-pipeline-with-pip-audit-and-sigstore/
├── lab-8.1-build-a-great-expectations-validation-suite-for-a-training-dataset/
├── lab-8.2-dvc-data-versioning-and-sha-256-dataset-hash-verification/
├── lab-8.3-add-differential-privacy-to-a-pytorch-training-loop-with-opacus/
├── lab-9.1-detect-poisoned-samples-in-a-fine-tuning-dataset-with-cleanlab/
├── lab-9.2-audit-a-lora-fine-tuned-model-for-behaviour-drift/
├── lab-10.1-build-a-github-actions-ml-security-pipeline/
├── lab-10.2-production-data-drift-monitoring-with-evidently-ai/
└── lab-12.1-capstone-detect-and-remove-poison-from-a-production-ml-pipeline/
```

Each lab folder contains:
- `README.md` — objectives, prerequisites, step-by-step tasks, expected outputs, discussion questions
- `requirements.txt` — exact package versions
- Python scripts — fully runnable starter code

---

## License

MIT License — free to use, share, and adapt with attribution.

**Armaan Sidana | Nexus Security | 2026**
