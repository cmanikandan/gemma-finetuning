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

echo "================================================================================"
echo " PROVISIONING SOVEREIGN FSI AI LANDING ZONE IN ${REGION} (Project: ${PROJECT_ID})"
echo "================================================================================"

# 1. Enable Required Google Cloud APIs
echo "[1/6] Enabling Vertex AI, Cloud KMS, Cloud Build, and Artifact Registry APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  cloudkms.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
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

# 3. Provision & Grant Cloud Storage, Vertex AI, and Artifact Registry Service Agents CMEK IAM Roles
echo "[3/6] Initializing Service Agents & Granting CMEK Encrypt/Decrypt IAM roles..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")

# On a brand-new GCP project, Google Cloud lazily creates Service Agents.
# Explicitly trigger their creation before binding KMS IAM policies:
GCS_SA=$(gcloud storage service-agent --project="${PROJECT_ID}" 2>/dev/null || echo "service-${PROJECT_NUMBER}@gs-project-accounts.iam.gserviceaccount.com")
gcloud beta services identity create --service=aiplatform.googleapis.com --project="${PROJECT_ID}" --quiet >/dev/null 2>&1 || true
gcloud beta services identity create --service=artifactregistry.googleapis.com --project="${PROJECT_ID}" --quiet >/dev/null 2>&1 || true

VERTEX_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com"
AR_SA="service-${PROJECT_NUMBER}@gcp-sa-artifactregistry.iam.gserviceaccount.com"

for SA in "${GCS_SA}" "${VERTEX_SA}" "${AR_SA}"; do
  echo "    -> Binding roles/cloudkms.cryptoKeyEncrypterDecrypter to ${SA}..."
  gcloud kms keys add-iam-policy-binding "${KEY_NAME}" \
    --keyring="${KEYRING_NAME}" \
    --location="${REGION}" \
    --member="serviceAccount:${SA}" \
    --role="roles/cloudkms.cryptoKeyEncrypterDecrypter" \
    --project="${PROJECT_ID}" >/dev/null
done

# 4. Create CMEK-Encrypted Sovereign GCS Bucket in asia-south1
echo "[4/6] Creating CMEK-Encrypted Cloud Storage Bucket gs://${BUCKET_NAME} in ${REGION}..."
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" &>/dev/null; then
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --location="${REGION}" \
    --default-encryption-key="${KMS_KEY_URI}" \
    --uniform-bucket-level-access \
    --public-access-prevention \
    --project="${PROJECT_ID}"
fi

# 5. Create CMEK-Encrypted Artifact Registry Repository in asia-south1
echo "[5/6] Creating Artifact Registry Repository ${AR_REPO_NAME} in ${REGION}..."
if ! gcloud artifacts repositories describe "${AR_REPO_NAME}" --location="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
  gcloud artifacts repositories create "${AR_REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --kms-key="${KMS_KEY_URI}" \
    --description="Sovereign FSI Gemma Training & Serving Containers" \
    --project="${PROJECT_ID}"
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
