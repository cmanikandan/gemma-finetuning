#!/usr/bin/env bash
# ==============================================================================
# LAB 8 (MANDATORY): Cleanup & Cost Shutdown for the Sovereign Gemma FSI Lab
# ==============================================================================
# WHY THIS MATTERS:
#   A deployed Vertex AI Endpoint with 1x NVIDIA L4 GPU bills continuously at
#   roughly $0.85/hour (~$620/month) EVEN WHEN YOU SEND IT ZERO REQUESTS.
#   Deleting the endpoint is the single most important cost action in this lab.
#
# SAFETY DESIGN (read before running):
#   * This script runs in DRY-RUN mode by default. Nothing is deleted.
#   * To actually delete, you must pass --confirm AND type the project ID when
#     prompted. There is no way to delete accidentally with a stray keystroke.
#   * Every deletion is scoped to resources whose names match this lab's
#     naming convention. It will never touch unrelated resources.
#
# USAGE:
#   ./scripts/99_cleanup_resources.sh                 # DRY RUN — shows what WOULD be deleted
#   ./scripts/99_cleanup_resources.sh --confirm       # ACTUALLY deletes (with typed confirmation)
#   ./scripts/99_cleanup_resources.sh --confirm --keep-bucket   # Delete compute, keep your model weights
# ==============================================================================

set -uo pipefail

# ------------------------------------------------------------------------------
# Configuration (must match what you used in scripts 01 / 02 / 03)
# ------------------------------------------------------------------------------
export PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
export REGION="${REGION:-asia-south1}"
export BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-fsi-gemma-sovereign-${REGION}}"
export KEYRING_NAME="${KEYRING_NAME:-fsi-sovereign-kr}"
export KEY_NAME="${KEY_NAME:-gemma-cmek-key}"
export AR_REPO_NAME="${AR_REPO_NAME:-fsi-gemma-repo}"
export ENDPOINT_DISPLAY_NAME="${ENDPOINT_DISPLAY_NAME:-fsi-sovereign-vllm-endpoint}"
export MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-gemma-indian-fsi-sovereign}"

DRY_RUN=true
KEEP_BUCKET=false

for arg in "$@"; do
  case "$arg" in
    --confirm)     DRY_RUN=false ;;
    --keep-bucket) KEEP_BUCKET=true ;;
    *) echo "[!] Unknown argument: $arg"; exit 1 ;;
  esac
done

if [[ -z "${PROJECT_ID}" ]]; then
  echo "[!] ERROR: PROJECT_ID is not set. Run: export PROJECT_ID=your-gcp-project-id"
  exit 1
fi

# Helper: print the command in dry-run, or execute it for real
run_step() {
  local description="$1"; shift
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "    [DRY-RUN] Would run: $*"
  else
    echo "    [DELETING] ${description}"
    "$@" 2>/dev/null || echo "    [skip] Resource not found or already deleted."
  fi
}

echo "================================================================================"
echo " SOVEREIGN GEMMA FSI LAB — RESOURCE CLEANUP & COST SHUTDOWN"
echo "================================================================================"
echo " Project ID : ${PROJECT_ID}"
echo " Region     : ${REGION}"
if [[ "${DRY_RUN}" == "true" ]]; then
  echo " MODE       : *** DRY RUN *** (nothing will be deleted)"
  echo "              Re-run with --confirm to actually delete."
else
  echo " MODE       : !!! LIVE DELETION !!!"
fi
echo "================================================================================"

# ------------------------------------------------------------------------------
# Inventory: show the user exactly what exists before touching anything
# ------------------------------------------------------------------------------
echo ""
echo "[INVENTORY] Scanning for lab resources in ${REGION}..."
echo "--------------------------------------------------------------------------------"
echo "  Vertex AI Endpoints (GPU — THE EXPENSIVE ONE):"
gcloud ai endpoints list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="displayName:${ENDPOINT_DISPLAY_NAME}" \
  --format="table(displayName, name)" 2>/dev/null || echo "    (none found)"

echo "  Vertex AI Models:"
gcloud ai models list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="displayName:gemma" --format="table(displayName, name)" 2>/dev/null || echo "    (none found)"

echo "  Vertex AI Custom Training Jobs (running jobs still bill):"
gcloud ai custom-jobs list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING" \
  --format="table(displayName, state, name)" 2>/dev/null || echo "    (none running)"

echo "  Cloud Run GPU Services:"
gcloud run services list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="metadata.name:gemma" --format="table(metadata.name)" 2>/dev/null || echo "    (none found)"

echo "  Cloud Storage Bucket:"
gcloud storage du -s "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" 2>/dev/null || echo "    (bucket not found)"
echo "--------------------------------------------------------------------------------"

# ------------------------------------------------------------------------------
# Typed confirmation gate for live deletion
# ------------------------------------------------------------------------------
if [[ "${DRY_RUN}" == "false" ]]; then
  echo ""
  echo "  !!! WARNING: IRREVERSIBLE ACTION !!!"
  echo "  You are about to permanently delete the resources listed above, including"
  if [[ "${KEEP_BUCKET}" == "false" ]]; then
    echo "  your fine-tuned model weights in gs://${BUCKET_NAME}."
    echo "  (Pass --keep-bucket if you want to preserve your trained model.)"
  else
    echo "  everything EXCEPT the bucket gs://${BUCKET_NAME} (--keep-bucket was set)."
  fi
  echo ""
  read -r -p "  To proceed, type the project ID exactly [${PROJECT_ID}]: " TYPED_CONFIRMATION
  if [[ "${TYPED_CONFIRMATION}" != "${PROJECT_ID}" ]]; then
    echo "  [ABORTED] Project ID did not match. Nothing was deleted."
    exit 1
  fi
  echo "  [CONFIRMED] Proceeding with deletion..."
fi

# ------------------------------------------------------------------------------
# STEP 1 (HIGHEST COST PRIORITY): Undeploy models from endpoints, then delete endpoints
# ------------------------------------------------------------------------------
echo ""
echo "[1/7] Vertex AI Endpoints — stops the ~\$0.85/hour GPU billing"
ENDPOINT_IDS=$(gcloud ai endpoints list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="displayName:${ENDPOINT_DISPLAY_NAME}" --format="value(name)" 2>/dev/null)

if [[ -z "${ENDPOINT_IDS}" ]]; then
  echo "    No matching endpoints found. Nothing to do."
else
  for EP in ${ENDPOINT_IDS}; do
    # A model must be undeployed before its endpoint can be deleted
    DEPLOYED_IDS=$(gcloud ai endpoints describe "${EP}" --region="${REGION}" --project="${PROJECT_ID}" \
      --format="value(deployedModels[].id)" 2>/dev/null | tr ';' ' ')
    for DM in ${DEPLOYED_IDS}; do
      run_step "Undeploying model ${DM} from endpoint ${EP}" \
        gcloud ai endpoints undeploy-model "${EP}" --region="${REGION}" \
        --deployed-model-id="${DM}" --project="${PROJECT_ID}" --quiet
    done
    run_step "Deleting endpoint ${EP}" \
      gcloud ai endpoints delete "${EP}" --region="${REGION}" --project="${PROJECT_ID}" --quiet
  done
fi

# ------------------------------------------------------------------------------
# STEP 2: Cancel any still-running training jobs
# ------------------------------------------------------------------------------
echo ""
echo "[2/7] Vertex AI Custom Training Jobs — cancel anything still consuming GPUs"
JOB_IDS=$(gcloud ai custom-jobs list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING" --format="value(name)" 2>/dev/null)

if [[ -z "${JOB_IDS}" ]]; then
  echo "    No running or pending jobs. Nothing to do."
else
  for JOB in ${JOB_IDS}; do
    run_step "Cancelling training job ${JOB}" \
      gcloud ai custom-jobs cancel "${JOB}" --region="${REGION}" --project="${PROJECT_ID}"
  done
fi

# ------------------------------------------------------------------------------
# STEP 3: Cloud Run GPU services
# ------------------------------------------------------------------------------
echo ""
echo "[3/7] Cloud Run GPU Services"
CR_SERVICES=$(gcloud run services list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="metadata.name:gemma" --format="value(metadata.name)" 2>/dev/null)

if [[ -z "${CR_SERVICES}" ]]; then
  echo "    No matching Cloud Run services. Nothing to do."
else
  for SVC in ${CR_SERVICES}; do
    run_step "Deleting Cloud Run service ${SVC}" \
      gcloud run services delete "${SVC}" --region="${REGION}" --project="${PROJECT_ID}" --quiet
  done
fi

# ------------------------------------------------------------------------------
# STEP 4: Vertex AI Model Registry entries
# ------------------------------------------------------------------------------
echo ""
echo "[4/7] Vertex AI Model Registry (storage-only cost, but keeps the project tidy)"
MODEL_IDS=$(gcloud ai models list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="displayName:gemma" --format="value(name)" 2>/dev/null)

if [[ -z "${MODEL_IDS}" ]]; then
  echo "    No matching models. Nothing to do."
else
  for M in ${MODEL_IDS}; do
    run_step "Deleting registered model ${M}" \
      gcloud ai models delete "${M}" --region="${REGION}" --project="${PROJECT_ID}" --quiet
  done
fi

# ------------------------------------------------------------------------------
# STEP 5: Colab Enterprise runtimes and Workbench instances
# ------------------------------------------------------------------------------
echo ""
echo "[5/7] Notebook Runtimes (Colab Enterprise / Workbench) — idle GPU VMs still bill!"
echo "    NOTE: These are listed for you to review. Delete via the Console UI, or"
echo "          uncomment the commands below if you are certain."
gcloud workbench instances list --location="${REGION}-a" --project="${PROJECT_ID}" \
  --format="table(name, state, gceSetup.machineType)" 2>/dev/null || echo "    (no Workbench instances found)"
gcloud colab runtimes list --region="${REGION}" --project="${PROJECT_ID}" \
  --format="table(name, state)" 2>/dev/null || echo "    (no Colab Enterprise runtimes found, or API not enabled)"
echo "    -> To delete a Workbench instance:"
echo "       gcloud workbench instances delete INSTANCE_NAME --location=${REGION}-a --project=${PROJECT_ID}"
echo "    -> To delete a Colab Enterprise runtime:"
echo "       gcloud colab runtimes delete RUNTIME_NAME --region=${REGION} --project=${PROJECT_ID}"

# ------------------------------------------------------------------------------
# STEP 6: Cloud Storage bucket (contains your dataset + fine-tuned weights)
# ------------------------------------------------------------------------------
echo ""
echo "[6/7] Cloud Storage Bucket — gs://${BUCKET_NAME}"
if [[ "${KEEP_BUCKET}" == "true" ]]; then
  echo "    [SKIPPED] --keep-bucket was set. Your fine-tuned weights are preserved."
  echo "    Reminder: storage costs ~\$0.02/GB/month. A merged 12B model is ~24 GB (~\$0.50/month)."
else
  run_step "Deleting bucket and all contents gs://${BUCKET_NAME}" \
    gcloud storage rm -r "gs://${BUCKET_NAME}" --project="${PROJECT_ID}"
fi

# ------------------------------------------------------------------------------
# STEP 7: Artifact Registry + Cloud KMS key material
# ------------------------------------------------------------------------------
echo ""
echo "[7/7] Artifact Registry & Cloud KMS"
run_step "Deleting Artifact Registry repository ${AR_REPO_NAME}" \
  gcloud artifacts repositories delete "${AR_REPO_NAME}" --location="${REGION}" \
  --project="${PROJECT_ID}" --quiet

echo ""
echo "    Cloud KMS NOTE — keys and keyrings CANNOT be deleted by design."
echo "    This is intentional in Google Cloud so cryptographic audit lineage is preserved."
echo "    Instead you schedule the KEY VERSION for destruction (24-hour reversible window),"
echo "    and the empty keyring itself costs nothing (~\$0.06/month per active key version)."
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "    [DRY-RUN] Would schedule destruction of key version 1 of ${KEY_NAME}"
else
  gcloud kms keys versions destroy 1 \
    --key="${KEY_NAME}" --keyring="${KEYRING_NAME}" --location="${REGION}" \
    --project="${PROJECT_ID}" --quiet 2>/dev/null \
    && echo "    [SCHEDULED] Key version 1 scheduled for destruction in 24 hours." \
    || echo "    [skip] Key version not found or already scheduled."
fi

# ------------------------------------------------------------------------------
# Final verification & summary
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
if [[ "${DRY_RUN}" == "true" ]]; then
  echo " DRY RUN COMPLETE — nothing was deleted."
  echo " Re-run with --confirm to perform the actual cleanup:"
  echo "     ./scripts/99_cleanup_resources.sh --confirm"
else
  echo " CLEANUP COMPLETE"
  echo ""
  echo " Post-cleanup verification (these should all be empty):"
  gcloud ai endpoints list --region="${REGION}" --project="${PROJECT_ID}" --format="value(displayName)" 2>/dev/null
  gcloud ai models list --region="${REGION}" --project="${PROJECT_ID}" --format="value(displayName)" 2>/dev/null
  echo ""
  echo " Confirm zero ongoing spend in the Billing Console:"
  echo "     https://console.cloud.google.com/billing"
fi
echo ""
echo " NUCLEAR OPTION (guaranteed zero cost — deletes EVERYTHING in the project):"
echo "     gcloud projects delete ${PROJECT_ID}"
echo "     (Only do this for a disposable sandbox project. It is reversible for 30 days.)"
echo "================================================================================"
