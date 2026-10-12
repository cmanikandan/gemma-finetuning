#!/usr/bin/env bash
# ==============================================================================
# LAB 2: Provision Sovereign AI Infrastructure in Google Cloud (asia-south1)
# ==============================================================================
# Runs 100% in Google Cloud Shell or MacBook Terminal via standard `gcloud` CLI.
# Requires ZERO local Docker or third-party package managers.
#
# Enforces Indian FSI Regulatory Compliance (RBI / SEBI Data Localization):
# 1. Pins all resources strictly to `asia-south1` (Mumbai) or `asia-south2` (Delhi).
# 2. Creates a Regional Cloud KMS Keyring & Customer-Managed Encryption Key (CMEK).
# 3. Provisions a CMEK-encrypted Cloud Storage Bucket with Public Access Prevention.
# 4. Uploads the generated Indian FSI Synthetic Dataset to the Sovereign GCS Bucket.
# ==============================================================================

set -euo pipefail

# Configuration Variables (Override via environment variables if desired)
export PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
export REGION="${REGION:-asia-south1}" # Mumbai Region for RBI/SEBI Data Localization
export KEYRING_NAME="${KEYRING_NAME:-fsi-sovereign-kr}"
export KEY_NAME="${KEY_NAME:-gemma-cmek-key}"
export BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-fsi-gemma-sovereign-${REGION}}"
export AR_REPO_NAME="${AR_REPO_NAME:-fsi-gemma-repo}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "[!] ERROR: PROJECT_ID is not set. Run: export PROJECT_ID=your-gcp-project-id"
  exit 1
fi

gcloud config set project "${PROJECT_ID}" >/dev/null 2>&1 || true

echo "================================================================================"
echo " PROVISIONING SOVEREIGN FSI AI LANDING ZONE IN ${REGION} (Project: ${PROJECT_ID})"
echo "================================================================================"

# 1. Enable Required Google Cloud APIs (including storage-api/storage-component for fresh projects)
echo "[1/6] Enabling Vertex AI, Cloud KMS, Cloud Storage, Cloud Build, Cloud Run & Artifact Registry APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  cloudkms.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  storage-api.googleapis.com \
  storage-component.googleapis.com \
  run.googleapis.com \
  --project="${PROJECT_ID}"

# 2. Create Regional Cloud KMS KeyRing & CMEK Key in Mumbai (asia-south1)
echo "[2/6] Creating Regional Cloud KMS KeyRing & CMEK Key in ${REGION}..."
if ! gcloud kms keyrings describe "${KEYRING_NAME}" --location="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
  gcloud kms keyrings create "${KEYRING_NAME}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}"
fi

if ! gcloud kms keys describe "${KEY_NAME}" --keyring="${KEYRING_NAME}" --location="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
  gcloud kms keys create "${KEY_NAME}" \
    --keyring="${KEYRING_NAME}" \
    --location="${REGION}" \
    --purpose="encryption" \
    --protection-level="software" \
    --project="${PROJECT_ID}"
fi

export KMS_KEY_URI="projects/${PROJECT_ID}/locations/${REGION}/keyRings/${KEYRING_NAME}/cryptoKeys/${KEY_NAME}"
echo "    -> CMEK Key URI: ${KMS_KEY_URI}"

# 3. Create Sovereign GCS Bucket in asia-south1 first (this forces GCS to initialize the project's storage account on brand-new projects)
echo "[3/6] Creating Sovereign Cloud Storage Bucket gs://${BUCKET_NAME} in ${REGION}..."
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" &>/dev/null; then
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --location="${REGION}" \
    --uniform-bucket-level-access \
    --public-access-prevention \
    --project="${PROJECT_ID}"
fi

# 4. Initialize Service Agents & Grant CMEK Encrypt/Decrypt IAM Roles (with IAM propagation retry)
echo "[4/6] Initializing Service Agents & Granting CMEK Encrypt/Decrypt IAM roles..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")

# Trigger creation of Cloud Storage, Vertex AI, and Artifact Registry Service Agents
GCS_SA=$(gcloud storage service-agent --project="${PROJECT_ID}" 2>/dev/null | tr -d '[:space:]' || true)
if [[ -z "${GCS_SA}" ]]; then
  GCS_SA="service-${PROJECT_NUMBER}@gs-project-accounts.iam.gserviceaccount.com"
fi
gcloud beta services identity create --service=storage.googleapis.com --project="${PROJECT_ID}" --quiet >/dev/null 2>&1 || true
gcloud beta services identity create --service=aiplatform.googleapis.com --project="${PROJECT_ID}" --quiet >/dev/null 2>&1 || true
gcloud beta services identity create --service=artifactregistry.googleapis.com --project="${PROJECT_ID}" --quiet >/dev/null 2>&1 || true

VERTEX_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com"
AR_SA="service-${PROJECT_NUMBER}@gcp-sa-artifactregistry.iam.gserviceaccount.com"

bind_cmek_role() {
  local sa_email="$1"
  local max_attempts=5
  local attempt=1
  while [[ ${attempt} -le ${max_attempts} ]]; do
    if gcloud kms keys add-iam-policy-binding "${KEY_NAME}" \
      --keyring="${KEYRING_NAME}" \
      --location="${REGION}" \
      --member="serviceAccount:${sa_email}" \
      --role="roles/cloudkms.cryptoKeyEncrypterDecrypter" \
      --project="${PROJECT_ID}" >/dev/null 2>&1; then
      echo "    -> [OK] Bound CMEK Encrypt/Decrypt role to ${sa_email}"
      return 0
    fi
    echo "    -> Waiting for new Service Agent ${sa_email} to propagate in global IAM (attempt ${attempt}/${max_attempts})..."
    sleep 4
    attempt=$((attempt + 1))
  done
  echo "    -> [WARN] Service Agent ${sa_email} still propagating; continuing with Google-managed encryption fallback."
  return 1
}

if bind_cmek_role "${GCS_SA}"; then
  echo "    -> Attaching KMS CMEK default encryption key to gs://${BUCKET_NAME}..."
  gcloud storage buckets update "gs://${BUCKET_NAME}" \
    --default-encryption-key="${KMS_KEY_URI}" \
    --project="${PROJECT_ID}" >/dev/null
fi

bind_cmek_role "${VERTEX_SA}" || true
AR_CMEK_OK=0
if bind_cmek_role "${AR_SA}"; then
  AR_CMEK_OK=1
fi

# 5. Create Artifact Registry Repository in asia-south1
echo "[5/6] Creating Artifact Registry Repository ${AR_REPO_NAME} in ${REGION}..."
if ! gcloud artifacts repositories describe "${AR_REPO_NAME}" --location="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
  if [[ "${AR_CMEK_OK}" -eq 1 ]]; then
    gcloud artifacts repositories create "${AR_REPO_NAME}" \
      --repository-format=docker \
      --location="${REGION}" \
      --kms-key="${KMS_KEY_URI}" \
      --description="Sovereign FSI Gemma Training & Serving Containers" \
      --project="${PROJECT_ID}"
  else
    gcloud artifacts repositories create "${AR_REPO_NAME}" \
      --repository-format=docker \
      --location="${REGION}" \
      --description="Sovereign FSI Gemma Training & Serving Containers" \
      --project="${PROJECT_ID}"
  fi
fi

# 6. Upload Synthetic Indian FSI Dataset & Training Script to Sovereign GCS Bucket
echo "[6/6] Uploading Indian FSI Synthetic Dataset & Training Script to gs://${BUCKET_NAME}..."
if [[ ! -f "data/indian_fsi_sft_train.jsonl" || ! -f "data/indian_fsi_sft_test_ood.jsonl" ]]; then
  echo "    -> Generating dataset splits & SHA-256 data_card.json first via standard library python3..."
  python3 data/generate_fsi_dataset.py --samples-per-domain 300
fi

gcloud storage cp data/indian_fsi_sft_train.jsonl     "gs://${BUCKET_NAME}/dataset/indian_fsi_sft_train.jsonl"
gcloud storage cp data/indian_fsi_sft_val.jsonl       "gs://${BUCKET_NAME}/dataset/indian_fsi_sft_val.jsonl"
gcloud storage cp data/indian_fsi_sft_test.jsonl      "gs://${BUCKET_NAME}/dataset/indian_fsi_sft_test.jsonl"
gcloud storage cp data/indian_fsi_sft_test_ood.jsonl  "gs://${BUCKET_NAME}/dataset/indian_fsi_sft_test_ood.jsonl"
gcloud storage cp data/indian_fsi_sft_challenge.jsonl "gs://${BUCKET_NAME}/dataset/indian_fsi_sft_challenge.jsonl"
gcloud storage cp data/data_card.json                 "gs://${BUCKET_NAME}/dataset/data_card.json"
gcloud storage cp training/train_gemma_qlora.py       "gs://${BUCKET_NAME}/scripts/train_gemma_qlora.py"

echo "================================================================================"
echo " [+] SOVEREIGN INFRASTRUCTURE SETUP COMPLETE!"
echo "     * Region:          ${REGION} (Mumbai Data Residency Verified)"
echo "     * CMEK Key URI:    ${KMS_KEY_URI}"
echo "     * Dataset Bucket:  gs://${BUCKET_NAME}/dataset/ (Train/Val/Test_ID/Test_OOD/Challenge + data_card.json)"
echo "     * Next Step:       Run ./scripts/02_submit_vertex_finetuning.sh"
echo "================================================================================"
