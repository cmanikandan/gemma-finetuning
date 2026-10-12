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

# Execution Mode: "prebuilt" (Default, instant launch) or "cloudbuild"
export BUILD_MODE="${BUILD_MODE:-prebuilt}"

# Hardware Configuration:
# g2-standard-12 provides 1x NVIDIA L4 (24GB VRAM) — ideal & cost-effective for 4-bit QLoRA on Gemma-2-9B / Gemma-3-12B
# For Gemma-2-27B or multi-GPU training, use g2-standard-48 (4x L4) or a2-highgpu-1g (1x A100 40GB)
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
export BASE_MIRROR="gs://${BUCKET_NAME}/models/base-mirror/${BASE_MODEL_ID//\//_}"
export JOB_NAME="gemma4-fsi-qlora-sft-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_GCS_URI="gs://${BUCKET_NAME}/models/${JOB_NAME}"

# First run pulls from Hugging Face Hub and mirrors to Mumbai GCS; subsequent runs load directly from the Mumbai mirror!
if gcloud storage ls "${BASE_MIRROR}/config.json" --project="${PROJECT_ID}" &>/dev/null; then
  echo "[*] Found in-region Mumbai GCS mirror for ${BASE_MODEL_ID}: ${BASE_MIRROR}"
  export EFFECTIVE_MODEL_ARG="${BASE_MIRROR}"
  export MIRROR_TO_ARG=""
else
  export EFFECTIVE_MODEL_ARG="${BASE_MODEL_ID}"
  export MIRROR_TO_ARG="${BASE_MIRROR}"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[!] WARNING: HF_TOKEN environment variable is not set."
  echo "    If pulling Gemma weights from HuggingFace Hub on the first run, export your token:"
  echo "    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx"
  echo "    (Subsequent runs load automatically from your Mumbai GCS mirror: ${BASE_MIRROR})"
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

if [[ "${BUILD_MODE}" == "cloudbuild" ]]; then
  echo "[*] Mode: Cloud Build — Building custom container remotely in Google Cloud (${REGION})..."
  export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/fsi-gemma-repo/gemma-qlora-trainer:v1"
  gcloud builds submit ./training \
    --region="${REGION}" \
    --tag="${IMAGE_URI}" \
    --project="${PROJECT_ID}"

  # Create Vertex AI Custom Job spec with custom image
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
      env:
        - name: HF_TOKEN
          value: "${HF_TOKEN:-}"
      args:
        - "--model-id=${EFFECTIVE_MODEL_ARG}"
        - "--mirror-to=${MIRROR_TO_ARG}"
        - "--train-dataset=gs://${BUCKET_NAME}/dataset/indian_fsi_sft_train.jsonl"
        - "--val-dataset=gs://${BUCKET_NAME}/dataset/indian_fsi_sft_val.jsonl"
        - "--output-dir=${OUTPUT_GCS_URI}"
        - "--lora-r=16"
        - "--lora-alpha=32"
        - "--epochs=3"
        - "--batch-size=2"
        - "--grad-accum=8"
        - "--merge-weights"
encryptionSpec:
  kmsKeyName: ${KMS_KEY_URI}
EOF

else
  echo "[*] Mode: Zero-Build Pre-Built PyTorch GPU Container (Instant Launch)..."
  # Uses Vertex AI Official PyTorch 2.4 CUDA 12.1 Container
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
      env:
        - name: HF_TOKEN
          value: "${HF_TOKEN:-}"
      command:
        - bash
        - -c
      args:
        - |
          set -e
          echo "[*] Installing / upgrading QLoRA & HuggingFace TRL dependencies inside ephemeral GPU VM..."
          pip install --no-cache-dir --upgrade "transformers>=4.46.0" "accelerate>=1.0.0" "peft>=0.13.0" "trl>=0.11.4" "bitsandbytes>=0.44.0" "datasets>=3.0.0" sentencepiece protobuf
          echo "[*] Downloading training script from Sovereign GCS Bucket..."
          gcloud storage cp gs://${BUCKET_NAME}/scripts/train_gemma_qlora.py /tmp/train_gemma_qlora.py
          echo "[*] Launching 4-Bit NF4 QLoRA Fine-Tuning on Gemma 4..."
          python3 /tmp/train_gemma_qlora.py \
            --model-id="${EFFECTIVE_MODEL_ARG}" \
            --mirror-to="${MIRROR_TO_ARG}" \
            --train-dataset="gs://${BUCKET_NAME}/dataset/indian_fsi_sft_train.jsonl" \
            --val-dataset="gs://${BUCKET_NAME}/dataset/indian_fsi_sft_val.jsonl" \
            --output-dir="${OUTPUT_GCS_URI}" \
            --lora-r=16 \
            --lora-alpha=32 \
            --epochs=3 \
            --batch-size=2 \
            --grad-accum=8 \
            --merge-weights
encryptionSpec:
  kmsKeyName: ${KMS_KEY_URI}
EOF
fi

echo "[*] Submitting Custom Training Job to Vertex AI (${REGION})..."
if ! gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --config="/tmp/vertex_custom_job_spec.yaml" \
  --project="${PROJECT_ID}" 2>/tmp/vertex_submit_err.log; then
  cat /tmp/vertex_submit_err.log
  if grep -qiE "kms|cryptoKey|serviceAccount|permission" /tmp/vertex_submit_err.log; then
    echo "[*] New project Vertex AI Custom Code Service Agent is still propagating for boot-disk CMEK."
    echo "    Retrying submission using CMEK-encrypted GCS bucket (${OUTPUT_GCS_URI}) with standard boot disk encryption..."
    sed -i.bak '/^encryptionSpec:/d; /kmsKeyName:/d' /tmp/vertex_custom_job_spec.yaml
    gcloud ai custom-jobs create \
      --region="${REGION}" \
      --display-name="${JOB_NAME}" \
      --config="/tmp/vertex_custom_job_spec.yaml" \
      --project="${PROJECT_ID}"
  else
    exit 1
  fi
fi

echo "================================================================================"
echo " [+] JOB SUBMITTED SUCCESSFULLY!"
echo "     Monitor live GPU logs in Cloud Shell or Console:"
echo "     gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT_ID}"
echo "================================================================================"
