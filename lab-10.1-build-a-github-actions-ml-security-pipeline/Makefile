# Makefile — ML Security Pipeline (Lab 10.1)
# Usage: make <target>

PYTHON      := python3
MODEL_FILE  := model.pt
SIG_FILE    := model.sig
DATA_CSV    := data/train.csv

# ---------------------------------------------------------------------------
# Local testing targets
# ---------------------------------------------------------------------------

.PHONY: test-local
test-local: $(DATA_CSV)
	@echo "==> Running data validation …"
	$(PYTHON) scripts/validate_data.py
	@echo "==> Running data integrity check …"
	$(PYTHON) scripts/check_data_integrity.py

.PHONY: scan
scan: $(MODEL_FILE)
	@echo "==> Running ModelScan …"
	$(PYTHON) scripts/scan_model.py
	@echo "==> Running Neural Cleanse check …"
	$(PYTHON) scripts/run_neural_cleanse.py

.PHONY: audit
audit:
	@echo "==> Running pip-audit …"
	pip-audit --output json --format json -o audit_results.json || true
	@echo "==> Checking audit results …"
	$(PYTHON) scripts/check_audit_results.py
	@echo "==> Verifying model signature …"
	$(PYTHON) scripts/verify_model_signature.py

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

$(DATA_CSV):
	@echo "==> Generating sample training data …"
	$(PYTHON) data/generate_sample_data.py

.PHONY: data
data: $(DATA_CSV)

# ---------------------------------------------------------------------------
# Model artifact creation and signing
# ---------------------------------------------------------------------------

$(MODEL_FILE):
	@echo "==> Creating dummy model artifact …"
	$(PYTHON) -c "\
import torch, torch.nn as nn; \
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2)); \
torch.save(model.state_dict(), '$(MODEL_FILE)'); \
print('[Makefile] $(MODEL_FILE) created.')"

.PHONY: sign-model
sign-model: $(MODEL_FILE)
	@echo "==> Signing $(MODEL_FILE) with HMAC-SHA256 …"
	$(PYTHON) -c "\
import hmac, hashlib, os, pathlib; \
secret_hex = os.environ.get('MODEL_HMAC_SECRET', '0' * 64); \
secret = bytes.fromhex(secret_hex); \
mac = hmac.new(secret, digestmod=hashlib.sha256); \
data = pathlib.Path('$(MODEL_FILE)').read_bytes(); \
mac.update(data); \
sig = mac.hexdigest(); \
pathlib.Path('$(SIG_FILE)').write_text(sig); \
print(f'[Makefile] Signature written to $(SIG_FILE): {sig}')"

# ---------------------------------------------------------------------------
# Full local pipeline
# ---------------------------------------------------------------------------

.PHONY: all
all: data $(MODEL_FILE) sign-model test-local scan audit
	@echo "==> Full local pipeline complete."

# ---------------------------------------------------------------------------
# Clean generated files
# ---------------------------------------------------------------------------

.PHONY: clean
clean:
	rm -f audit_results.json modelscan_report.json neural_cleanse_report.json \
	      ge_validation_result.json signature_verification.log \
	      $(SIG_FILE)
	@echo "==> Cleaned generated artefacts."
