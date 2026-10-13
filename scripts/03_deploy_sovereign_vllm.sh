#!/usr/bin/env bash
# ==============================================================================
# LAB 5: Deploy Fine-Tuned FSI Gemma Model to Sovereign vLLM Endpoint (asia-south1)
# ==============================================================================
# Deploys the merged BF16 fine-tuned Gemma model (or base + LoRA adapter) to:
#   Option A: Vertex AI Dedicated Prediction Endpoint (with CMEK & Private VPC)
#   Option B: Cloud Run GPU (Serverless NVIDIA L4 in asia-south1 with IAM Auth)
# ==============================================================================

set -euo pipefail

export PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
export REGION="${REGION:-asia-south1}"
export BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-fsi-gemma-sovereign-${REGION}}"
export DEPLOY_TARGET="${DEPLOY_TARGET:-vertex-endpoint}" # "vertex-endpoint" or "cloud-run-gpu"

# ------------------------------------------------------------------------------
# Auto-discover the merged model produced by Lab 3 (scripts/02_submit_vertex_finetuning.sh).
# That script writes to: gs://BUCKET/models/gemma-fsi-qlora-sft-<TIMESTAMP>/merged_model_bf16
# We pick the most recent one automatically so you never have to paste a timestamp.
# Override by exporting MODEL_ARTIFACT_GCS_URI yourself.
# ------------------------------------------------------------------------------
if [[ -z "${MODEL_ARTIFACT_GCS_URI:-}" ]]; then
  echo "[*] Auto-discovering the most recent completed fine-tuned model in gs://${BUCKET_NAME}/models/ ..."
  LATEST_CONFIG=$(gcloud storage ls "gs://${BUCKET_NAME}/models/*/merged_model_bf16/config.json" --project="${PROJECT_ID}" 2>/dev/null | sort | tail -n 1 || true)
  if [[ -n "${LATEST_CONFIG}" ]]; then
    export MODEL_ARTIFACT_GCS_URI="${LATEST_CONFIG%/config.json}"
  else
    LATEST_RUN=$(gcloud storage ls "gs://${BUCKET_NAME}/models/" --project="${PROJECT_ID}" 2>/dev/null | grep -E "sft-" | sort | tail -n 1 || true)
    if [[ -z "${LATEST_RUN}" ]]; then
      echo "[!] ERROR: No fine-tuned models found in gs://${BUCKET_NAME}/models/"
      echo "    Run ./scripts/02_submit_vertex_finetuning.sh first and wait for it to finish, OR"
      echo "    export MODEL_ARTIFACT_GCS_URI=gs://your-bucket/path/to/merged_model_bf16"
      exit 1
    fi
    export MODEL_ARTIFACT_GCS_URI="${LATEST_RUN%/}/merged_model_bf16"
  fi
  echo "    -> Found: ${MODEL_ARTIFACT_GCS_URI}"
fi

# Verify the merged weights actually exist before burning 20 minutes on a failed deploy
if ! gcloud storage ls "${MODEL_ARTIFACT_GCS_URI}/config.json" --project="${PROJECT_ID}" &>/dev/null; then
  echo "[!] ERROR: No config.json found at ${MODEL_ARTIFACT_GCS_URI}"
  echo "    The training job may still be running, or it was run without the --merge-weights flag."
  echo "    Check job status: gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT_ID}"
  exit 1
fi

export MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-gemma4-indian-fsi-sovereign}"
export ENDPOINT_DISPLAY_NAME="${ENDPOINT_DISPLAY_NAME:-fsi-sovereign-vllm-endpoint}"

# Official Google Cloud Vertex AI vLLM GPU Serving Container for Gemma 4
export VLLM_IMAGE_URI="${VLLM_IMAGE_URI:-us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:gemma4}"

echo "================================================================================"
echo " DEPLOYING SOVEREIGN FSI GEMMA MODEL VIA vLLM IN ${REGION}"
echo "================================================================================"
echo " * Target Infrastructure : ${DEPLOY_TARGET}"
echo " * Model Artifacts URI   : ${MODEL_ARTIFACT_GCS_URI}"
echo " * Serving Container     : vLLM PagedAttention GPU Engine"
echo "================================================================================"

if [[ "${DEPLOY_TARGET}" == "cloud-run-gpu" ]]; then
  echo "[*] Deploying Serverless Scale-to-Zero Cloud Run GPU Service (NVIDIA L4) in ${REGION}..."
  # Compute relative path inside bucket for native Cloud Run GCS FUSE volume mount at /models
  RELATIVE_MODEL_PATH="${MODEL_ARTIFACT_GCS_URI#gs://${BUCKET_NAME}/}"
  MOUNTED_MODEL_PATH="/models/${RELATIVE_MODEL_PATH}"

  VLLM_ARGS=(
    serve "${MOUNTED_MODEL_PATH}"
    --served-model-name="${MODEL_DISPLAY_NAME}"
    --max-model-len=4096
    --max-num-seqs=16
    --gpu-memory-utilization=0.90
    --dtype=bfloat16
    --quantization=bitsandbytes
    --load-format=bitsandbytes
    --enable-prefix-caching
    '--limit-mm-per-prompt={"image":0,"audio":0}'
    --generation-config=auto
    --host=0.0.0.0 --port=8080
  )
  # Join args with '@' delimiter so gcloud does not split the JSON string in --limit-mm-per-prompt
  JOINED_ARGS="$(IFS=@; echo "${VLLM_ARGS[*]}")"

  gcloud beta run deploy "${MODEL_DISPLAY_NAME}" \
    --image="${VLLM_IMAGE_URI}" \
    --region="${REGION}" \
    --execution-environment=gen2 \
    --cpu=8 \
    --memory=32Gi \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --no-gpu-zonal-redundancy \
    --no-cpu-throttling \
    --min-instances=0 \
    --max-instances=2 \
    --concurrency=16 \
    --timeout=300 \
    --no-allow-unauthenticated \
    --add-volume=name=models,type=cloud-storage,bucket="${BUCKET_NAME}",readonly=true \
    --add-volume-mount=volume=models,mount-path=/models \
    --startup-probe=tcpSocket.port=8080,initialDelaySeconds=240,failureThreshold=3,timeoutSeconds=240,periodSeconds=240 \
    --command=vllm \
    --args="^@^${JOINED_ARGS}" \
    --project="${PROJECT_ID}"

  echo "[+] Cloud Run GPU Sovereign Service Deployed (Scale-to-Zero + Prefix Caching Enabled)!"
  echo "    Test via local proxy: gcloud run services proxy ${MODEL_DISPLAY_NAME} --region=${REGION} --port=9090"
elif [[ "${DEPLOY_TARGET}" == "model-garden" ]]; then
  echo "[*] Deploying Custom Weights directly via Vertex AI Model Garden in ${REGION}..."
  gcloud ai model-garden models deploy \
    --model="${MODEL_ARTIFACT_GCS_URI}" \
    --region="${REGION}" \
    --machine-type=g2-standard-12 \
    --accelerator-type=NVIDIA_L4 \
    --accelerator-count=1 \
    --project="${PROJECT_ID}"
  echo "[+] Vertex AI Model Garden Custom Weights Deployment Initiated!"
else
  # Ensure the Vertex AI Service Agent & Compute Engine Default SA can read the merged model from the CMEK-encrypted bucket
  PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
  VERTEX_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com"
  CUSTOM_CODE_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"
  COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
  for sa in "${VERTEX_SA}" "${CUSTOM_CODE_SA}" "${COMPUTE_SA}"; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
      --member="serviceAccount:${sa}" \
      --role="roles/storage.objectViewer" \
      --condition=None --quiet >/dev/null 2>&1 || true
    gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
      --member="serviceAccount:${sa}" \
      --role="roles/storage.objectViewer" \
      --project="${PROJECT_ID}" >/dev/null 2>&1 || true
    gcloud kms keys add-iam-policy-binding "gemma-cmek-key" \
      --keyring="fsi-sovereign-kr" \
      --location="${REGION}" \
      --member="serviceAccount:${sa}" \
      --role="roles/cloudkms.cryptoKeyEncrypterDecrypter" \
      --project="${PROJECT_ID}" >/dev/null 2>&1 || true
  done
  gcloud iam service-accounts add-iam-policy-binding "${COMPUTE_SA}" \
    --member="serviceAccount:${VERTEX_SA}" \
    --role="roles/iam.serviceAccountUser" \
    --project="${PROJECT_ID}" >/dev/null 2>&1 || true

  echo "[*] Syncing training/serve_gemma_vertex.py to gs://${BUCKET_NAME}/scripts/serve_gemma_vertex.py..."
  gcloud storage cp ./training/serve_gemma_vertex.py "gs://${BUCKET_NAME}/scripts/serve_gemma_vertex.py" --project="${PROJECT_ID}" >/dev/null

  # Prefer the verified in-region Gemma 4 GPU container in Artifact Registry if built during Step 4
  TRAINER_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/fsi-gemma-repo/gemma-qlora-trainer:v1"
  if gcloud artifacts docker images describe "${TRAINER_IMAGE_URI}" --project="${PROJECT_ID}" &>/dev/null; then
    SERVE_IMAGE_URI="${TRAINER_IMAGE_URI}"
    echo "[*] Using verified in-region Gemma 4 GPU container: ${SERVE_IMAGE_URI}"
  else
    SERVE_IMAGE_URI="${VLLM_IMAGE_URI}"
  fi

  # Base64-encode bootstrap so --container-args has zero spaces, commas, or shell-escaping ambiguities
  BOOTSTRAP_B64=$(python3 -c "
import base64
code = '''import json, os, urllib.request
req = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token', headers={'Metadata-Flavor': 'Google'})
tok = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())['access_token']
url = 'https://storage.googleapis.com/storage/v1/b/${BUCKET_NAME}/o/scripts%2Fserve_gemma_vertex.py?alt=media'
r = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + tok})
with urllib.request.urlopen(r, timeout=30) as resp, open('/tmp/serve_gemma_vertex.py', 'wb') as f:
    f.write(resp.read())
os.execvp('python3', ['python3', '/tmp/serve_gemma_vertex.py'])
'''
print(base64.b64encode(code.encode()).decode())
")

  echo "[1/3] Uploading Fine-Tuned Gemma Model to Vertex AI Model Registry (${REGION})..."
  gcloud ai models upload \
    --region="${REGION}" \
    --display-name="${MODEL_DISPLAY_NAME}" \
    --container-image-uri="${SERVE_IMAGE_URI}" \
    --artifact-uri="${MODEL_ARTIFACT_GCS_URI}" \
    --container-command="python3" \
    --container-args="^|^-c|import base64;exec(base64.b64decode('${BOOTSTRAP_B64}'))" \
    --container-ports=8080 \
    --container-predict-route="/predict" \
    --container-health-route="/health" \
    --project="${PROJECT_ID}" >/dev/null

  MODEL_RESOURCE=$(gcloud ai models list \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --filter="displayName=${MODEL_DISPLAY_NAME}" \
    --sort-by="~createTime" \
    --format="value(name)" 2>/dev/null | head -n 1)
  echo "    -> Uploaded Model Resource: ${MODEL_RESOURCE}"

  echo "[2/3] Locating or Creating Dedicated Vertex AI Prediction Endpoint in ${REGION}..."
  ENDPOINT_RESOURCE=$(gcloud ai endpoints list \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --filter="displayName=${ENDPOINT_DISPLAY_NAME}" \
    --sort-by="~createTime" \
    --format="value(name)" 2>/dev/null | head -n 1 || true)

  if [[ -z "${ENDPOINT_RESOURCE}" ]]; then
    ENDPOINT_RESOURCE=$(gcloud ai endpoints create \
      --region="${REGION}" \
      --display-name="${ENDPOINT_DISPLAY_NAME}" \
      --project="${PROJECT_ID}" \
      --format="value(name)")
  fi
  echo "    -> Endpoint Resource: ${ENDPOINT_RESOURCE}"

  echo "[3/3] Deploying Model to Endpoint on NVIDIA L4 GPU (g2-standard-12) as ${COMPUTE_SA}..."
  echo "      (Note: Provisioning the GPU VM and loading the 12B checkpoint takes ~10-15 minutes)"
  gcloud ai endpoints deploy-model "${ENDPOINT_RESOURCE}" \
    --region="${REGION}" \
    --model="${MODEL_RESOURCE}" \
    --display-name="${MODEL_DISPLAY_NAME}-v1" \
    --machine-type="g2-standard-12" \
    --accelerator="type=nvidia-l4,count=1" \
    --service-account="${COMPUTE_SA}" \
    --min-replica-count=1 \
    --max-replica-count=1 \
    --project="${PROJECT_ID}"

  echo "[+] Vertex AI Dedicated GPU Endpoint Deployed in ${REGION}!"
fi
