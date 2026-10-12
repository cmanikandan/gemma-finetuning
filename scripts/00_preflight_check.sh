#!/usr/bin/env bash
# ==============================================================================
# LAB 0: Pre-Flight Check — Verify Your Environment Before Spending Any Money
# ==============================================================================
# Run this FIRST, in Google Cloud Shell or your terminal. It is 100% read-only:
# it creates nothing, deletes nothing, and costs nothing.
#
# It verifies the 7 things that cause 95% of failures later in the lab:
#   1. gcloud is installed and you are authenticated
#   2. A project is selected and billing is enabled
#   3. Required APIs are enabled
#   4. You have GPU quota in your chosen region
#   5. Your Hugging Face token is set (needed for gated Gemma weights)
#   6. Python 3 is available for the local data-generation steps
#   7. The repository files are all present
# ==============================================================================

set -uo pipefail

export PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
export REGION="${REGION:-asia-south1}"

PASS=0
WARN=0
FAIL=0

ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
warn() { echo "  [WARN] $1"; WARN=$((WARN+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "================================================================================"
echo " PRE-FLIGHT CHECK — SOVEREIGN GEMMA FSI FINE-TUNING LAB"
echo " Target Region: ${REGION}"
echo "================================================================================"

# ------------------------------------------------------------------------------
echo ""
echo "[1/7] gcloud CLI & Authentication"
if ! command -v gcloud &>/dev/null; then
  bad "gcloud CLI not found. Use Google Cloud Shell (it is pre-installed) or install the SDK."
else
  ok "gcloud CLI found: $(gcloud --version 2>/dev/null | head -n 1)"
  ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -n 1)
  if [[ -z "${ACTIVE_ACCOUNT}" ]]; then
    bad "No active gcloud account. Run: gcloud auth login"
  else
    ok "Active account: ${ACTIVE_ACCOUNT}"
  fi
  # Application Default Credentials are used by Python client libraries
  if gcloud auth application-default print-access-token &>/dev/null; then
    ok "Application Default Credentials (ADC) are configured."
  else
    warn "ADC not configured. If you run Python GCP libraries locally, run:"
    echo "         gcloud auth application-default login"
    echo "         gcloud auth application-default set-quota-project ${PROJECT_ID}"
  fi
fi

# ------------------------------------------------------------------------------
echo ""
echo "[2/7] Project & Billing"
if [[ -z "${PROJECT_ID}" ]]; then
  bad "No project selected. Run: gcloud config set project YOUR_PROJECT_ID"
else
  ok "Project: ${PROJECT_ID}"
  BILLING_ENABLED=$(gcloud beta billing projects describe "${PROJECT_ID}" \
    --format="value(billingEnabled)" 2>/dev/null)
  if [[ "${BILLING_ENABLED}" == "True" ]]; then
    ok "Billing is enabled on this project."
  else
    warn "Could not confirm billing is enabled (you may lack billing viewer permission)."
    echo "         GPUs will NOT provision without an active billing account."
    echo "         Verify at: https://console.cloud.google.com/billing/linkedaccount?project=${PROJECT_ID}"
  fi
fi

# ------------------------------------------------------------------------------
echo ""
echo "[3/7] Required Google Cloud APIs"
REQUIRED_APIS=(
  "aiplatform.googleapis.com"
  "storage.googleapis.com"
  "cloudkms.googleapis.com"
  "cloudbuild.googleapis.com"
  "artifactregistry.googleapis.com"
  "compute.googleapis.com"
)
ENABLED_APIS=$(gcloud services list --enabled --project="${PROJECT_ID}" --format="value(config.name)" 2>/dev/null)
MISSING_APIS=()
for API in "${REQUIRED_APIS[@]}"; do
  if echo "${ENABLED_APIS}" | grep -q "^${API}$"; then
    ok "${API}"
  else
    MISSING_APIS+=("${API}")
    warn "${API} is NOT enabled"
  fi
done
if [[ ${#MISSING_APIS[@]} -gt 0 ]]; then
  echo "       -> Enable them all with:"
  echo "          gcloud services enable ${MISSING_APIS[*]} --project=${PROJECT_ID}"
  echo "          (scripts/01_setup_sovereign_infra.sh also does this for you)"
fi

# ------------------------------------------------------------------------------
echo ""
echo "[4/7] GPU Quota in ${REGION} (the #1 cause of lab failure)"
L4_QUOTA=$(gcloud compute regions describe "${REGION}" --project="${PROJECT_ID}" \
  --format="value(quotas[].limit)" --flatten="quotas[]" \
  --filter="quotas.metric=NVIDIA_L4_GPUS" 2>/dev/null | head -n 1)
T4_QUOTA=$(gcloud compute regions describe "${REGION}" --project="${PROJECT_ID}" \
  --format="value(quotas[].limit)" --flatten="quotas[]" \
  --filter="quotas.metric=NVIDIA_T4_GPUS" 2>/dev/null | head -n 1)

if [[ -n "${L4_QUOTA}" && "${L4_QUOTA%.*}" -gt 0 ]]; then
  ok "NVIDIA L4 GPU quota in ${REGION}: ${L4_QUOTA%.*} (you need 1). Recommended path."
elif [[ -n "${T4_QUOTA}" && "${T4_QUOTA%.*}" -gt 0 ]]; then
  warn "No L4 quota, but NVIDIA T4 quota is ${T4_QUOTA%.*} in ${REGION}."
  echo "         -> Use a T4 and a smaller model: export BASE_MODEL_ID=google/gemma-3-4b-it"
  echo "            export MACHINE_TYPE=n1-standard-8 ACCELERATOR_TYPE=NVIDIA_TESLA_T4"
else
  warn "No GPU quota detected in ${REGION}. You have three options:"
  echo "         (a) Request quota: https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}"
  echo "             Filter for 'NVIDIA L4 GPUs' in ${REGION}, select it, and click EDIT QUOTAS."
  echo "         (b) Switch region:  export REGION=asia-southeast1   (or us-central1)"
  echo "         (c) Use Dynamic Workload Scheduler to queue until capacity frees up."
fi
echo "       Note: Vertex AI Custom Training uses a separate 'Custom model training' quota pool."
echo "       Check it at: https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}"

# ------------------------------------------------------------------------------
echo ""
echo "[5/7] Hugging Face Token (required to download gated Gemma weights)"
if [[ -n "${HF_TOKEN:-}" ]]; then
  ok "HF_TOKEN is set (${HF_TOKEN:0:7}...)"
else
  warn "HF_TOKEN is not set. Gemma weights are license-gated on Hugging Face."
  echo "         1. Visit https://huggingface.co/google/gemma-3-12b-it and accept the license."
  echo "         2. Create a READ token at https://huggingface.co/settings/tokens"
  echo "         3. Run: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx"
  echo "         (Alternatively, deploy Gemma from Vertex AI Model Garden, which needs no HF token.)"
fi

# ------------------------------------------------------------------------------
echo ""
echo "[6/7] Local Python (only needed for the free data-generation steps)"
if command -v python3 &>/dev/null; then
  ok "python3 found: $(python3 --version 2>&1)"
  echo "       Note: data/generate_fsi_dataset.py and data/inspect_dataset.py use ONLY the"
  echo "       Python standard library — no pip install, no venv, no uv required."
else
  bad "python3 not found. Google Cloud Shell has it pre-installed."
fi

# ------------------------------------------------------------------------------
echo ""
echo "[7/7] Repository Files"
REQUIRED_FILES=(
  "data/generate_fsi_dataset.py"
  "data/inspect_dataset.py"
  "training/train_gemma_qlora.py"
  "training/Dockerfile"
  "scripts/01_setup_sovereign_infra.sh"
  "scripts/02_submit_vertex_finetuning.sh"
  "scripts/03_deploy_sovereign_vllm.sh"
  "scripts/99_cleanup_resources.sh"
  "evaluation/evaluate_fsi_model.py"
  "evaluation/passbook_parser_benchmark.py"
  "evaluation/predict_endpoint.py"
  "evaluation/hillclimbing_simulator.py"
)
for F in "${REQUIRED_FILES[@]}"; do
  if [[ -f "${F}" ]]; then
    ok "${F}"
  else
    bad "MISSING: ${F} (are you running this from the repository root?)"
  fi
done

# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo " PRE-FLIGHT SUMMARY:  ${PASS} passed  |  ${WARN} warnings  |  ${FAIL} failures"
echo "================================================================================"
if [[ ${FAIL} -gt 0 ]]; then
  echo " ACTION REQUIRED: Resolve the [FAIL] items above before continuing."
  exit 1
elif [[ ${WARN} -gt 0 ]]; then
  echo " You can proceed, but review the [WARN] items — especially GPU quota and HF_TOKEN."
  echo " Next step: ./scripts/01_setup_sovereign_infra.sh"
  exit 0
else
  echo " All checks passed. You are ready to run the full lab."
  echo " Next step: ./scripts/01_setup_sovereign_infra.sh"
  exit 0
fi
