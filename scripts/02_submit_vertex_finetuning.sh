#!/usr/bin/env bash
# ==============================================================================
# LAB 3: Submit Sovereign Gemma QLoRA Fine-Tuning Job to Vertex AI (asia-south1)
# ==============================================================================
# Designed specifically for corporate MacBooks / Google Cloud Shell where local
# Docker daemon or `uv` installation is blocked by security policy.
#
# Supports TWO Zero-Local-Docker Execution Modes:
#   MODE 1 (DEFAULT - ZERO BUILD): Uses Google Cloud's Official Pre-Built PyTorch
#          GPU Deep Learning Container (`pytorch-gpu.2-4.py310`). Installs PEFT/TRL
#          at container startup and runs `train_gemma_qlora.py` directly from GCS.
#   MODE 2 (CLOUD BUILD REMOTE): Uses `gcloud builds submit` to build the custom
#          Dockerfile remotely inside Google Cloud (`asia-south1`) without touching
#          your local machine.
# ==============================================================================

set -euo pipefail

export PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
export REGION="${REGION:-asia-south1}"
export BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-fsi-gemma-sovereign-${REGION}}"
export KEYRING_NAME="${KEYRING_NAME:-fsi-sovereign-kr}"
export KEY_NAME="${KEY_NAME:-gemma-cmek-key}"
export KMS_KEY_URI="projects/${PROJECT_ID}/locations/${REGION}/keyRings/${KEYRING_NAME}/cryptoKeys/${KEY_NAME}"

# Execution Mode: "cloudbuild" (Default & required for Gemma 4: builds pytorch:2.13.0-cuda12.6 remotely in GCP)
export BUILD_MODE="${BUILD_MODE:-cloudbuild}"

# Hardware Configuration:
# g2-standard-12 provides 1x NVIDIA L4 (24GB VRAM, 12 vCPU, 48GB RAM) — ideal & cost-effective for 4-bit QLoRA on Gemma 4
export MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-12}"
export ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_L4}"
export ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"

# Supported Model Generations (Gemma 4 is the default):
# - Gemma 4 12B Unified (Default):       google/gemma-4-12B-it     (Single L4 24GB sweet spot; encoder-free unified model)
# - Gemma 4 E4B (Fast/Lower-Resource):   google/gemma-4-E4B-it     (Single L4 24GB; fastest training & lowest VRAM)
# - Gemma 4 MoE (26B Total / 4B Active): google/gemma-4-26B-A4B-it (Fits on 1x L4 24GB or 2x L4 with ultra-fast 4B active inference!)
# - Gemma 4 Dense Flagship (31B):        google/gemma-4-31B-it     (Use g2-standard-24 [2x L4] or a2-highgpu-1g [1x A100])
# - Legacy Gemma 3 / 2 (Supported):      google/gemma-3-12b-it, google/gemma-2-9b-it
export BASE_MODEL_ID="${BASE_MODEL_ID:-google/gemma-4-12B-it}"
export EPOCHS="${EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export BASE_MIRROR="gs://${BUCKET_NAME}/models/base-mirror/${BASE_MODEL_ID//\//_}"
export JOB_NAME="gemma4-fsi-qlora-sft-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_GCS_URI="gs://${BUCKET_NAME}/models/${JOB_NAME}"

# Gemma 4 requires Python 3.11 + PyTorch 2.13 (CUDA 12.6) + transformers==5.19.0 + trl==1.15.0.
# Vertex AI's legacy prebuilt container (pytorch-gpu.2-4.py310) has Python 3.10 + PyTorch 2.4,
# and upgrading torch at runtime pulls CUDA 13 wheels that fail on CUDA 12 host drivers.
if echo "${BASE_MODEL_ID}" | grep -qi "gemma-4" && [[ "${BUILD_MODE}" == "prebuilt" ]]; then
  echo "[*] Note: Gemma 4 (${BASE_MODEL_ID}) requires Python 3.11 + PyTorch 2.13 (CUDA 12.6) + transformers==5.19.0."
  echo "    Switching BUILD_MODE from 'prebuilt' (legacy Python 3.10 / PyTorch 2.4) to 'cloudbuild' (Zero Local Docker)."
  export BUILD_MODE="cloudbuild"
fi

# First run pulls from Hugging Face Hub and mirrors to Mumbai GCS; subsequent runs load directly from the Mumbai mirror!
if gcloud storage ls "${BASE_MIRROR}/config.json" --project="${PROJECT_ID}" &>/dev/null; then
  echo "[*] Found in-region Mumbai GCS mirror for ${BASE_MODEL_ID}: ${BASE_MIRROR}"
  export EFFECTIVE_MODEL_ARG="${BASE_MIRROR}"
  export MIRROR_TO_ARG=""
else
  export EFFECTIVE_MODEL_ARG="${BASE_MODEL_ID}"
  export MIRROR_TO_ARG="${BASE_MIRROR}"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  ENV_YAML="      env:
        - name: HF_TOKEN
          value: \"${HF_TOKEN}\""
else
  echo "[*] Note: HF_TOKEN is not set (not required for ungated Apache 2.0 Gemma 4 models or Mumbai GCS mirror)."
  ENV_YAML="      env:
        - name: PYTHONUNBUFFERED
          value: \"1\""
fi

echo "================================================================================"
echo " SUBMITTING VERTEX AI SOVEREIGN QLORA FINE-TUNING JOB"
echo "================================================================================"
echo " * Job Name:         ${JOB_NAME}"
echo " * Region:           ${REGION} (India Data Residency)"
echo " * Execution Mode:   ${BUILD_MODE} (Zero Local Docker)"
echo " * Compute Node:     ${MACHINE_TYPE} (${ACCELERATOR_COUNT}x ${ACCELERATOR_TYPE})"
echo " * Base Model:       ${EFFECTIVE_MODEL_ARG}"
echo " * In-Region Mirror: ${BASE_MIRROR}"
echo " * Output Artifacts: ${OUTPUT_GCS_URI}"
echo "================================================================================"

echo "[*] Syncing latest training/train_gemma_qlora.py to gs://${BUCKET_NAME}/scripts/train_gemma_qlora.py..."
gcloud storage cp ./training/train_gemma_qlora.py "gs://${BUCKET_NAME}/scripts/train_gemma_qlora.py" --project="${PROJECT_ID}" >/dev/null

# Ensure Compute Engine Default SA & Vertex AI Custom Code Service Agent have GCS + CMEK + Artifact Registry permissions
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
CUSTOM_CODE_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"

for role in "roles/storage.objectAdmin" "roles/artifactregistry.writer" "roles/logging.logWriter"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="${role}" \
    --condition=None --quiet >/dev/null 2>&1 || true
done
for sa in "${COMPUTE_SA}" "${CUSTOM_CODE_SA}"; do
  gcloud kms keys add-iam-policy-binding "${KEY_NAME}" \
    --keyring="${KEYRING_NAME}" \
    --location="${REGION}" \
    --member="serviceAccount:${sa}" \
    --role="roles/cloudkms.cryptoKeyEncrypterDecrypter" \
    --project="${PROJECT_ID}" >/dev/null 2>&1 || true
done

if [[ "${BUILD_MODE}" == "cloudbuild" ]]; then
  export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/fsi-gemma-repo/gemma-qlora-trainer:v1"
  if [[ "${FORCE_REBUILD:-0}" != "1" ]] && gcloud artifacts docker images describe "${IMAGE_URI}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "[*] Found pre-built Gemma 4 container in Artifact Registry: ${IMAGE_URI}"
    echo "    Skipping Cloud Build for instant launch! (Export FORCE_REBUILD=1 to force a container rebuild)"
  else
    echo "[*] Mode: Cloud Build — Building Gemma 4 container (pytorch:2.13.0-cuda12.6 + transformers==5.19.0) remotely in ${REGION}..."
    gcloud builds submit ./training \
      --region="${REGION}" \
      --tag="${IMAGE_URI}" \
      --timeout=3600 \
      --project="${PROJECT_ID}"
  fi

  # Create Vertex AI CustomJobSpec YAML (CustomJobSpec accepts workerPoolSpecs & baseOutputDirectory)
  cat <<EOF > /tmp/vertex_custom_job_spec.yaml
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE_TYPE}
      acceleratorType: ${ACCELERATOR_TYPE}
      acceleratorCount: ${ACCELERATOR_COUNT}
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 200
    containerSpec:
      imageUri: ${IMAGE_URI}
${ENV_YAML}
      command:
        - python3
        - "/gcs/${BUCKET_NAME}/scripts/train_gemma_qlora.py"
      args:
        - "--model-id=${EFFECTIVE_MODEL_ARG}"
        - "--mirror-to=${MIRROR_TO_ARG}"
        - "--train-dataset=gs://${BUCKET_NAME}/dataset/indian_fsi_sft_train.jsonl"
        - "--val-dataset=gs://${BUCKET_NAME}/dataset/indian_fsi_sft_val.jsonl"
        - "--output-dir=${OUTPUT_GCS_URI}"
        - "--lora-r=16"
        - "--lora-alpha=32"
        - "--epochs=${EPOCHS}"
        - "--max-steps=${MAX_STEPS}"
        - "--batch-size=2"
        - "--grad-accum=8"
        - "--merge-weights"
baseOutputDirectory:
  outputUriPrefix: ${OUTPUT_GCS_URI}
EOF

else
  echo "[*] Mode: Zero-Build Pre-Built PyTorch GPU Container (Legacy Gemma 2 / 3)..."
  export PREBUILT_IMAGE="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"

  cat <<EOF > /tmp/vertex_custom_job_spec.yaml
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE_TYPE}
      acceleratorType: ${ACCELERATOR_TYPE}
      acceleratorCount: ${ACCELERATOR_COUNT}
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 200
    containerSpec:
      imageUri: ${PREBUILT_IMAGE}
${ENV_YAML}
      command:
        - bash
        - -c
      args:
        - |
          set -e
          export TMPDIR=/var/tmp
          mkdir -p /var/tmp
          echo "[*] Installing QLoRA & HuggingFace TRL stack inside ephemeral GPU VM..."
          pip install --no-cache-dir --upgrade "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "transformers>=5.10.1" "accelerate>=1.1.0" "peft>=0.19.0" "trl>=1.14.0" "bitsandbytes>=0.48.0" "datasets>=3.0.0" sentencepiece protobuf
          python3 -c "import torch, transformers, trl, peft, bitsandbytes; print('[+] Verified runtime stack:', torch.__version__, transformers.__version__, trl.__version__, peft.__version__)"
          echo "[*] Downloading training script from Sovereign GCS Bucket..."
          gcloud storage cp gs://${BUCKET_NAME}/scripts/train_gemma_qlora.py /tmp/train_gemma_qlora.py
          echo "[*] Launching 4-Bit NF4 QLoRA Fine-Tuning..."
          python3 /tmp/train_gemma_qlora.py \
            --model-id="${EFFECTIVE_MODEL_ARG}" \
            --mirror-to="${MIRROR_TO_ARG}" \
            --train-dataset="gs://${BUCKET_NAME}/dataset/indian_fsi_sft_train.jsonl" \
            --val-dataset="gs://${BUCKET_NAME}/dataset/indian_fsi_sft_val.jsonl" \
            --output-dir="${OUTPUT_GCS_URI}" \
            --lora-r=16 \
            --lora-alpha=32 \
            --epochs=${EPOCHS} \
            --max-steps=${MAX_STEPS} \
            --batch-size=2 \
            --grad-accum=8 \
            --merge-weights
baseOutputDirectory:
  outputUriPrefix: ${OUTPUT_GCS_URI}
EOF
fi

echo "[*] Submitting Custom Training Job to Vertex AI (${REGION})..."
gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --config="/tmp/vertex_custom_job_spec.yaml" \
  --project="${PROJECT_ID}"

JOB_RESOURCE=$(gcloud ai custom-jobs list \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --filter="displayName=${JOB_NAME}" \
  --format="value(name)" | head -n 1)

echo "================================================================================"
echo " [+] JOB SUBMITTED SUCCESSFULLY!"
echo "     * Job Resource : ${JOB_RESOURCE:-${JOB_NAME}}"
echo "     * Output URI   : ${OUTPUT_GCS_URI} (CMEK Encrypted in ${REGION})"
echo "     * Stream Logs  : gcloud ai custom-jobs stream-logs ${JOB_RESOURCE:-<JOB_ID>} --region=${REGION}"
echo "     * List Jobs    : gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT_ID}"
echo "================================================================================"
