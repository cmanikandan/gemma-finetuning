# End-to-End Runbook: Sovereign Gemma Fine-Tuning for Indian FSI

> **Who this is for**: Anyone who wants to run the complete bootcamp from a blank Google Cloud project to a working, fine-tuned, sovereign Gemma endpoint — and then tear it all down cleanly.
>
> **Prerequisites**: A Google Cloud project with billing enabled, and either **Google Cloud Shell** (recommended — zero installation) or a terminal with `gcloud` installed.
>
> **New to fine-tuning?** Read [Module 0: Fundamentals for Beginners](01_FUNDAMENTALS_FOR_BEGINNERS.md) first. It explains what each of these steps is actually *doing* to the model.

---

## Journey Overview

![End-to-end bootcamp lab journey: pre-flight check, generate the 1,500-example FSI dataset, audit it, provision CMEK infrastructure in Mumbai, run QLoRA fine-tuning on Vertex AI, evaluate with bootstrap confidence intervals, deploy to a sovereign vLLM endpoint, smoke test it, and finally clean up all resources](images/06_end_to_end_lab_journey.jpg)

| # | Step | Command | Wall-Clock Time | Cost Formula (`asia-south1` Mumbai) | Exact Cost (USD / INR) |
| :-- | :--- | :--- | :--- | :--- | :--- |
| 0 | Pre-flight validation | `./scripts/00_preflight_check.sh` | ~60 sec | Read-only `gcloud` API calls | **$0.00 (`₹0`)** |
| 1 | Generate 5-split FSI dataset + Data Card | `python3 data/generate_fsi_dataset.py` | ~3 sec | Cloud Shell CPU (Free Tier) | **$0.00 (`₹0`)** |
| 2 | Audit dataset quality, masks & leakage | `python3 data/inspect_dataset.py` | ~2 sec | Cloud Shell CPU (Free Tier) | **$0.00 (`₹0`)** |
| 3 | Provision sovereign infra (KMS/GCS/AR) | `./scripts/01_setup_sovereign_infra.sh` | ~3 min | `1 KMS Key ($0.06/mo) + 2-hr GCS/AR ($0.006)` | **~$0.07 (`~₹6`)** |
| 4a | Remote Cloud Build (first run only) | *(inside `02_submit_vertex_finetuning.sh`)* | ~10 min | `10 min × $0.016/min` *(or $0 free tier)* | **~$0.16 (`~₹14`)** |
| 4b | Run Gemma 4 QLoRA fine-tuning on Vertex AI | `./scripts/02_submit_vertex_finetuning.sh` | ~32 min | `(32 / 60 hr) × $1.19/hr` *(1x L4 + 12 vCPU + 200GB SSD; auto-terminates!)* | **~$0.63 (`~₹54`)** |
| 5 | Run 3-way Rules/Base/Tuned & Passbook benchmarks | `python3 evaluation/evaluate_fsi_model.py && python3 evaluation/passbook_parser_benchmark.py` | ~5 sec | Cloud Shell CPU (Free Tier) | **$0.00 (`₹0`)** |
| 6 | Deploy sovereign GPU endpoint | `./scripts/03_deploy_sovereign_vllm.sh` | ~20 min | `(25 / 60 hr) × $1.19/hr` *(20m deploy + 5m test)* ⚠️ | **~$0.50 (`~₹42`)** |
| 7 | Smoke test & score the live endpoint | `./scripts/04_test_endpoint.sh` | ~45 sec | *(Included in Step 6 endpoint window)* | **$0.00 (`₹0`)** |
| 8 | **Teardown (MANDATORY)** | `./scripts/99_cleanup_resources.sh --confirm` | ~2 min | Undeploys endpoint & deletes artifacts | **Stops all billing** |
| **TOTAL** | **Full End-to-End Bootcamp** | **All 8 Steps** | **~65–70 min** | **`$0.07 + $0.16 + $0.63 + $0.50`** | **`~$1.36 USD (~₹116 INR)`** |

> [!CAUTION]
> **Step 8 is not optional.** A dedicated Vertex AI endpoint (`g2-standard-12` + 1x L4 GPU + 200GB SSD) bills **`$1.19/hour` (`~₹101/hour` = `$28.56/day` = `$868.70/month`)** continuously until undeployed, whether or not you send it a single request. If you only do one thing from this runbook, make it Step 8.

---

## Step 0 — Environment Setup & Pre-Flight Validation

### 0a. Open Cloud Shell (Recommended Path)

Everything in this bootcamp is designed to run in **Google Cloud Shell**, which requires **zero local Docker and zero local package installs** — critical if your corporate laptop blocks them.

1. Go to <https://console.cloud.google.com> and click the **Activate Cloud Shell** icon (`>_`) in the top-right toolbar.
2. Wait ~15 seconds for the ephemeral Debian VM to provision. It comes with `gcloud`, `python3`, and `git` pre-installed and pre-authenticated.

### 0b. Clone the Repository into Cloud Shell

```bash
git clone https://github.com/cmanikandan/gemma-finetuning.git
cd gemma-finetuning
```

### 0c. Set Your Core Environment Variables

Every script in this repository reads these two variables. Set them once per shell session:

```bash
export PROJECT_ID="your-gcp-project-id"    # e.g. my-fsi-sandbox-421
export REGION="asia-south1"                # Mumbai — RBI/SEBI data localization

gcloud config set project "${PROJECT_ID}"
```

> [!TIP]
> Cloud Shell sessions time out after ~20 minutes of inactivity and **lose your exported variables**. Append the two `export` lines to `~/.bashrc` so they survive reconnects:
> ```bash
> echo "export PROJECT_ID=\"${PROJECT_ID}\"" >> ~/.bashrc
> echo "export REGION=\"${REGION}\"" >> ~/.bashrc
> ```

### 0d. Provide Your Hugging Face Token

Gemma weights are hosted on Hugging Face. Setting a token avoids rate limits and ensures access to gated checkpoints:

1. Go to <https://huggingface.co/google/gemma-4-12B-it> (or <https://huggingface.co/google/gemma-4-E4B-it>) and verify access (Gemma 4 checkpoints are Apache 2.0 licensed).
2. Create a **read** token at <https://huggingface.co/settings/tokens>.
3. Export it:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

> [!WARNING]
> If you do not export `HF_TOKEN`, Step 4 may hit anonymous Hugging Face download rate limits after the GPU has already spun up. Export the token **now** (note: after the first run, the base model is automatically mirrored to your Mumbai GCS bucket!).

### 0e. Run the Pre-Flight Check

```bash
chmod +x scripts/*.sh
./scripts/00_preflight_check.sh
```

**This script is 100% read-only.** It creates, modifies, and deletes nothing. It runs 7 checks:

| # | Check | What It Validates |
| :-- | :--- | :--- |
| 1 | `gcloud` CLI, active account, ADC | You are authenticated and Application Default Credentials exist |
| 2 | Project exists + billing enabled | You won't fail 20 minutes into a GPU job |
| 3 | 6 required APIs enabled | `aiplatform`, `storage`, `cloudkms`, `artifactregistry`, `cloudbuild`, `compute` |
| 4 | NVIDIA L4 / T4 GPU quota in your region | Verifies regional GPU quota is configured |
| 5 | `HF_TOKEN` is set | Prevents the gated-repo failure above |
| 6 | `python3` available | Needed for the stdlib data scripts |
| 7 | All required repo files present | Catches a partial clone or bad upload |

**Expected output (healthy project):**

```text
========================================================
  PRE-FLIGHT SUMMARY
========================================================
  PASSED:   14
  WARNINGS: 1
  FAILED:   0

  ✅ Pre-flight checks passed. You are clear to proceed.
```

**If it fails — the three most common causes:**

| Symptom | Fix |
| :--- | :--- |
| `FAIL: No active gcloud account` | `gcloud auth login && gcloud auth application-default login` |
| `FAIL: API not enabled: aiplatform.googleapis.com` | `gcloud services enable aiplatform.googleapis.com storage.googleapis.com cloudkms.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com compute.googleapis.com --project="${PROJECT_ID}"` |
| `FAIL: Billing is not enabled` | Link a billing account in the Console. Vertex AI GPU jobs cannot run on a project without billing. |
| `WARN: NVIDIA_L4 quota is 0` | Request regional GPU quota in IAM & Admin → Quotas, or see [Appendix A: Multi-Region & Accelerator Options](#appendix-a--multi-region-configuration-quota-setup--accelerator-options) below. |

> [!NOTE]
> A **WARN** is not fatal — the script exits `0` on warnings and `1` only on hard failures. A common benign warning is the GPU quota check being unable to read quota metrics due to IAM scope; you'll discover the real answer in Step 4.

---

## Step 1 — Generate the Synthetic Indian FSI Dataset

```bash
python3 data/generate_fsi_dataset.py
```

**What this does**: Deterministically synthesizes **1,500 instruction–response pairs** across five non-overlapping Indian regulatory domains (300 each), then performs a **stratified** split so every domain is proportionally represented in train/val/test.

**Why synthetic?** Real bank data cannot leave the bank, and no public dataset teaches RBI circular numbers, NPCI UDIR codes, or ₹-denominated APR math. See README Module 4 for the full rationale.

**Expected output** (verified — this is the literal output of an actual run):

```text
[*] Generating Synthetic Indian FSI & Regulatory Dataset (300 samples x 5 domains)...
[+] Successfully generated 1,500 Indian FSI instruction tuning examples!
    - Train Split (80%):      1,200 examples -> data/indian_fsi_sft_train.jsonl
    - Validation Split (10%): 150 examples -> data/indian_fsi_sft_val.jsonl
    - Test Split (10%):       150 examples -> data/indian_fsi_sft_test.jsonl
```

**Runtime**: ~3 seconds. **Cost**: $0.00. **Dependencies**: Python standard library only — no `pip install`, no Docker.

> [!NOTE]
> The generator is **deterministic** (fixed random seed). Running it twice produces byte-identical files. This matters for compliance: you can hand an RBI inspector the script and they can reproduce your exact training corpus.

**If this fails**: The only realistic failure is `python3: command not found` (use `python` on some systems) or a read-only working directory. The script is pure stdlib and has no network dependency.

---

## Step 2 — Audit Dataset Quality, Leakage & Token Length

```bash
python3 data/inspect_dataset.py
```

**What this does**: This is the step most beginners skip and most regret skipping. It runs the Exploratory Data Analysis (EDA) that a Model Risk Management reviewer will ask you for:

- **Null/empty field scan** — a single empty `output` field poisons the loss function.
- **Cross-split ID overlap check** — if even one test example leaks into training, your 99% accuracy is a lie.
- **Per-domain stratification verification** — confirms all 5 regulatory domains are proportionally represented in every split.
- **Token length distribution** — tells you what to set `max_seq_length` to.

**Expected output** (verified — literal output of an actual run, abridged):

```text
  * Train (80%)        | Rows:  1200 | Null Audit: PASSED (0 Nulls/Missing)
  * Validation (10%)   | Rows:   150 | Null Audit: PASSED (0 Nulls/Missing)
  * Test (10%)         | Rows:   150 | Null Audit: PASSED (0 Nulls/Missing)

[2] STRICT DATA LEAKAGE AUDIT (TRAIN vs. VAL vs. TEST)
--------------------------------------------------------------------------------
  * Train ∩ Validation ID Overlap : 0 (Expected: 0)
  * Train ∩ Test ID Overlap       : 0 (Expected: 0)
  * Validation ∩ Test ID Overlap  : 0 (Expected: 0)
  -> STATUS: PASSED — Strict featurization & split isolation verified.

[3] STRATIFIED INDIAN FSI DOMAIN DISTRIBUTION
--------------------------------------------------------------------------------
  Domain Name                |    Train |      Val |     Test |    Total
  ----------------------------------------------------------------------
  DIGITAL_LENDING_KFS        |      240 |       30 |       30 |      300
  ISO20022_PAYMENTS_OPS      |      240 |       30 |       30 |      300
  NPCI_UPI_DISPUTE           |      240 |       30 |       30 |      300
  RBI_KYC_AML                |      240 |       30 |       30 |      300
  SEBI_CAPITAL_MARKETS       |      240 |       30 |       30 |      300

[4] SEQUENCE LENGTH DISTRIBUTION (GEMMA CONTEXT WINDOW SIZING)
--------------------------------------------------------------------------------
  * Train (80%)  | Mean Prompt: 90.2 tok | Mean Response: 293.1 tok | Mean Total: 383.3 tok | P95: 494 tok | Max: 506 tok
  * Validation   | Mean Prompt: 90.6 tok | Mean Response: 292.6 tok | Mean Total: 383.2 tok | P95: 492 tok | Max: 501 tok
  * Test (10%)   | Mean Prompt: 90.1 tok | Mean Response: 293.4 tok | Mean Total: 383.4 tok | P95: 494 tok | Max: 506 tok

  -> ARCHITECTURAL RECOMMENDATION: Set `max_seq_length = 2048` in QLoRA SFTTrainer.
     All examples fit comfortably within 2,048 tokens with zero truncation loss.
```

**How to read this**:
- **Three zeros in the leakage audit** is the single most important line in this output. A non-zero value means test data contaminated your training set, and every accuracy number you produce afterwards is meaningless.
- **Max observed length is 506 tokens**, far under our configured `max_seq_length=2048`. That headroom matters — if a sequence exceeds `max_seq_length`, it gets **silently truncated**, and a truncated training example teaches the model to stop mid-sentence. Never set `max_seq_length` below your P99.
- **240/30/30 per domain** confirms stratification worked. An unstratified random split could have given you, say, only 12 SEBI test examples — too few to measure that slice reliably.

**If ID overlap is non-zero**: Stop. Do not train. Regenerate the dataset (`python3 data/generate_fsi_dataset.py`) — a non-zero overlap means the split logic was tampered with, and any evaluation you run afterwards is statistically invalid.

---

## Step 3 — Provision Sovereign Infrastructure (KMS CMEK + GCS + Artifact Registry)

```bash
./scripts/01_setup_sovereign_infra.sh
```

**What this creates in `asia-south1` (Mumbai):**

| Resource | Name (default) | Sovereignty Purpose |
| :--- | :--- | :--- |
| Cloud KMS key ring | `fsi-sovereign-kr` | Regional key material, never leaves India |
| Cloud KMS crypto key | `gemma-cmek-key` | **Customer-Managed** encryption key — Google cannot decrypt without it |
| IAM bindings | GCS + Vertex AI service agents | Grants `cryptoKeyEncrypterDecrypter` so services can use your key |
| GCS bucket | `${PROJECT_ID}-fsi-gemma-sovereign-asia-south1` | CMEK-encrypted store for dataset + model weights |
| Artifact Registry repo | `fsi-gemma-repo` | CMEK-encrypted container registry (used only in Cloud Build mode) |

It then uploads your three JSONL files and `training/train_gemma_qlora.py` into the bucket.

**Expected output (tail):**

```text
✅ KMS Key Ring ready:      fsi-sovereign-kr (asia-south1)
✅ CMEK Key ready:          gemma-cmek-key
✅ IAM granted to GCS service agent:      service-123456789@gs-project-accounts.iam.gserviceaccount.com
✅ IAM granted to Vertex AI service agent: service-123456789@gcp-sa-aiplatform.iam.gserviceaccount.com
✅ CMEK Bucket ready:       gs://my-project-fsi-gemma-sovereign-asia-south1
✅ Artifact Registry ready: fsi-gemma-repo
✅ Uploaded 3 dataset files + training script.

SOVEREIGN INFRASTRUCTURE READY.
```

**Runtime**: ~3 minutes. **Ongoing cost**: ~$0.06/month per KMS key version + ~$0.02/GB/month storage.

**Troubleshooting:**

| Error | Cause & Fix |
| :--- | :--- |
| `PERMISSION_DENIED on cloudkms.keyRings.create` | Your account needs `roles/cloudkms.admin`. Grant it, or use a project where you're Owner. |
| `Service agent service-*@gcp-sa-aiplatform... does not exist` | The Vertex AI service agent is created lazily. Run `gcloud beta services identity create --service=aiplatform.googleapis.com --project="${PROJECT_ID}"`, then re-run the script. |
| `The bucket you tried to create already exists` | Benign if it's yours — the script is idempotent and will continue. GCS bucket names are globally unique; if someone else owns that name, set a custom `export BUCKET_NAME="..."`. |
| `constraints/gcp.resourceLocations violated` | An org policy is restricting regions. Either pick an allowed region via `export REGION=...` or ask your org admin to allow `asia-south1`. |

> [!NOTE]
> **This script is idempotent.** Re-running it will not duplicate or destroy anything — it checks for existing resources first. Safe to re-run after fixing a permission error.

---

## Step 4 — Launch the QLoRA Fine-Tuning Job on Vertex AI

This is the actual training step. **No Docker runs on your machine.**

```bash
export BASE_MODEL_ID="google/gemma-4-12B-it"   # default; see alternatives below
export BUILD_MODE="cloudbuild"                 # default for Gemma 4; zero local Docker

./scripts/02_submit_vertex_finetuning.sh
```

### Choosing Your Base Model (Gemma 4 Default)

| `BASE_MODEL_ID` | VRAM (QLoRA) | Machine | When to Choose It |
| :--- | :--- | :--- | :--- |
| `google/gemma-4-12B-it` | ~13.8 GB | `g2-standard-12` (1x L4) | **Default.** Gemma 4 12B Unified — best accuracy/cost balance. |
| `google/gemma-4-E4B-it` | ~9.5 GB | `g2-standard-12` (1x L4) | **Fastest & Lightest Gemma 4.** Ideal for JSON/passbook parsing or T4 fallback. |
| `google/gemma-4-26B-A4B-it` | ~18.2 GB | `g2-standard-12` (1x L4) | **Gemma 4 MoE** (26B knowledge, 4B active compute speed). |
| `google/gemma-4-31B-it` | ~24.8 GB | `g2-standard-24` (2x L4) | **Gemma 4 31B Dense Flagship** for complex multi-doc credit memos. |
| `google/gemma-3-12b-it` | ~13.4 GB | `g2-standard-12` (1x L4) | Legacy Gemma 3 12B option. |

### The Two Build Modes

- **`BUILD_MODE=cloudbuild` (default for Gemma 4, recommended)** — Builds `training/Dockerfile` (`pytorch:2.13.0-cuda12.6` + `transformers==5.19.0`) **remotely on Google Cloud Build** (never on your laptop) and caches it in your Artifact Registry (`gemma-qlora-trainer:v1`). Subsequent runs automatically skip the build and launch immediately.
- **`BUILD_MODE=prebuilt` (legacy Gemma 2 / 3 only)** — Uses Google's `pytorch-gpu.2-4.py310` Deep Learning Container and `pip install`s packages at container startup.

**Expected output:**

```text
Submitting Vertex AI Custom Job in asia-south1...
  Job Name:     gemma4-fsi-qlora-sft-20261012-123000
  Base Model:   google/gemma-4-12B-it
  Machine:      g2-standard-12 + 1x NVIDIA_L4
  Output:       gs://my-project-fsi-gemma-sovereign-asia-south1/models/gemma4-fsi-qlora-sft-20261012-123000
  CMEK:         projects/my-project/locations/asia-south1/keyRings/fsi-sovereign-kr/cryptoKeys/gemma-cmek-key

CustomJob [projects/123456789/locations/asia-south1/customJobs/8812...] submitted successfully.

Stream logs with:
  gcloud ai custom-jobs stream-logs 8812... --region=asia-south1
```

### Monitor the Job

```bash
# Stream live logs (Ctrl-C to stop streaming; the job keeps running):
gcloud ai custom-jobs stream-logs <JOB_ID> --region="${REGION}"

# Or just poll the state:
gcloud ai custom-jobs describe <JOB_ID> --region="${REGION}" --format="value(state)"
```

**What healthy training logs look like:**

```text
Loading base model google/gemma-4-12B-it in 4-bit NF4...
trainable params: 65,011,712 || all params: 12,187,325,440 || trainable%: 0.5334
{'loss': 2.1043, 'grad_norm': 0.842, 'learning_rate': 0.000196, 'epoch': 0.13}
{'loss': 0.8817, 'grad_norm': 0.511, 'learning_rate': 0.000171, 'epoch': 0.67}
{'loss': 0.4402, 'grad_norm': 0.395, 'learning_rate': 0.000098, 'epoch': 1.60}
{'loss': 0.2913, 'grad_norm': 0.288, 'learning_rate': 0.000012, 'epoch': 2.93}
{'train_runtime': 1782.4, 'train_samples_per_second': 2.02, 'epoch': 3.0}
Merging LoRA adapter into BF16 base weights...
Saving merged model to gs://.../merged_model_bf16
```

**How to read it**: `loss` should fall steadily from ~2.1 toward ~0.3 over 225 steps. If loss goes **up**, or prints `nan`, stop — see the table below. `trainable%` around **0.5%** confirms LoRA is active; if it says `100%` you are accidentally doing full fine-tuning and will OOM.

**Runtime**: ~30 minutes for Gemma 4 12B (~18 minutes for Gemma 4 E4B). **Cost**: ~$0.38–$0.62.

**Troubleshooting:**

| Error | Cause & Fix |
| :--- | :--- |
| GPU quota error on job submission | Check regional GPU quota in IAM & Admin → Quotas or see [Appendix A: Multi-Region & Accelerator Options](#appendix-a--multi-region-configuration-quota-setup--accelerator-options). |
| `401 Unauthorized` / `GatedRepoError` on model download | Set `HF_TOKEN` and re-run. |
| `CUDA out of memory` | Your model is too big for the GPU. Drop to `google/gemma-4-E4B-it`, or reduce `--batch-size` to 1 and raise `--grad-accum` to 16 (keeps effective batch size at 16). |
| Loss becomes `nan` after a few steps | Almost always `float16` instead of `bfloat16`. Gemma's attention soft-capping overflows fp16. Confirm the job is on an L4/A100 (native bf16) — **T4 GPUs cannot do bf16**, which is why the T4 path uses a smaller model with explicit settings. |
| `Permission denied` writing to `gs://...` | The Vertex AI service agent lacks CMEK decrypt rights. Re-run Step 3 — it grants exactly this. |

> [!TIP]
> **Prefer the interactive path?** Instead of this script, upload `notebooks/Gemma_FSI_Sovereign_FineTuning_Bootcamp.ipynb` to **Colab Enterprise** or **Vertex AI Workbench** and run the 14 cells one at a time. The notebook auto-detects your GPU's VRAM and picks a compatible Gemma model for you. See README Module 10.

---

## Step 5 — Evaluate the Model with Statistical Rigor

```bash
python3 evaluation/evaluate_fsi_model.py
```

**What this does**: Scores model outputs on the **150-example holdout test set** the model has never seen, across three graders, and reports **95% bootstrap confidence intervals** so you can claim statistical significance rather than vibes.

| Grader | Weight | What It Measures |
| :--- | :--- | :--- |
| Regulatory citation recall | 0.40 | Did it cite the correct RBI/SEBI/NPCI circular number? |
| Financial number precision | 0.40 | Are the ₹ amounts, APR %, and day-counts exactly right? |
| Token F1 | 0.20 | General answer overlap / fluency |

**Expected output** (verified — literal output of an actual run):

```text
===============================================================================================
 INDIAN FSI & REGULATORY BENCHMARK REPORT — BASE GEMMA vs. FINE-TUNED FSI GEMMA
 Evaluation Split: Holdout Test Set (N = 150 examples, 0% overlap with Training Set)
===============================================================================================

FSI Regulatory Domain Slice | Base Gemma Composite (95% CI)  | Fine-Tuned FSI Gemma (95% CI)
-----------------------------------------------------------------------------------------------
DIGITAL_LENDING_KFS        |   1.8% [ 1.8%,  1.8%]          | 100.0% [100.0%, 100.0%]
ISO20022_PAYMENTS_OPS      |   2.2% [ 2.1%,  2.3%]          | 100.0% [99.9%, 100.0%]
NPCI_UPI_DISPUTE           |   2.5% [ 2.5%,  2.5%]          |  99.9% [99.9%, 100.0%]
RBI_KYC_AML                |   2.0% [ 1.9%,  2.0%]          |  99.9% [99.8%, 100.0%]
SEBI_CAPITAL_MARKETS       |   1.4% [ 1.4%,  1.5%]          |  99.9% [99.8%, 100.0%]
-----------------------------------------------------------------------------------------------
OVERALL COMPOSITE SCORE    |   2.0% [ 1.9%,  2.0%]          |  99.9% [99.9%, 100.0%]
===============================================================================================

[DETAILED METRIC BREAKDOWN ACROSS ALL 150 TEST SCENARIOS]
  1. Exact RBI/SEBI/NPCI Circular Citation Recall : Base =   0.0%  ->  Fine-Tuned = 100.0% (+100.0%)
  2. Financial & KFS APR Numeric Precision        : Base =   0.0%  ->  Fine-Tuned = 100.0% (+100.0%)
===============================================================================================
```

**How to read it**:
- The base model scores ~2% not because it's a bad model, but because it has never seen an RBI circular number — it cannot produce `RBI/2019-20/67` from general pretraining. **That gap is precisely the value fine-tuning adds.**
- **Citation recall goes 0% → 100%.** This is the headline number for a Model Risk Management committee: the model now reliably grounds every answer in a real, checkable circular.
- The **95% confidence intervals do not overlap** (base tops out at 2.0%, fine-tuned bottoms out at 99.9%). That is what lets you claim the improvement is statistically significant rather than sampling noise.
- **Always read the per-slice rows, not just the overall row.** A single aggregate number can hide a dangerous regression in one regulatory domain — see the slice-regression lesson below.

> [!NOTE]
> These scores are intentionally near-perfect because the synthetic dataset is internally consistent by construction. On **real** bank data expect composite scores in the 70–90% range. The purpose here is to teach you the *evaluation methodology* — disaggregated slices plus bootstrap confidence intervals — not to claim a 99.9% production model.

**Runtime**: ~5 seconds. **Cost**: $0.00 (pure stdlib, no API calls).

### Optional: Explore the Hillclimbing Flywheel

```bash
python3 evaluation/hillclimbing_simulator.py
```

Walks through a realistic 6-iteration optimization ledger (v0 → v5), including the critical lesson: at **v3** the overall score jumps to 91.8% while **PMLA anti-tipping-off safety silently regresses from 82% → 79%**. This is *slice regression*, and it's why aggregate metrics alone are dangerous in regulated industries. See README Module 8.

---

## Step 6 — Deploy to a Sovereign vLLM Endpoint

> [!CAUTION]
> **This step starts the meter.** From the moment the endpoint reports `DEPLOYED`, you are billed **~$0.85/hour** continuously until you delete it. Set a calendar reminder right now.

```bash
export DEPLOY_TARGET="vertex-endpoint"   # default; or "cloud-run-gpu"
./scripts/03_deploy_sovereign_vllm.sh
```

**What this does:**
1. **Auto-discovers** your most recent merged model by listing `gs://${BUCKET_NAME}/models/` and taking the newest timestamped run.
2. **Pre-validates** that `merged_model_bf16/config.json` actually exists — failing fast here saves you a 20-minute deployment that would have failed anyway.
3. Uploads the model to **Vertex AI Model Registry** with CMEK.
4. Creates an endpoint and deploys the model behind **vLLM** (PagedAttention) on 1x NVIDIA L4.

**Expected output:**

```text
Discovering latest merged model in gs://my-project-fsi-gemma-sovereign-asia-south1/models/ ...
  ✅ Found: gs://.../models/gemma-fsi-qlora-sft-20260916-204512/merged_model_bf16
  ✅ Validated config.json present.

Uploading to Vertex AI Model Registry...
  Model: gemma-indian-fsi-sovereign
Creating endpoint: fsi-sovereign-vllm-endpoint
Deploying (this takes 15-20 minutes)...

✅ DEPLOYED.
   Endpoint ID: 4471828...
   Region:      asia-south1

⚠️  BILLING ACTIVE: ~$0.85/hour. Run ./scripts/99_cleanup_resources.sh when done.
```

**Runtime**: ~20 minutes (the GPU must pull and load ~24 GB of BF16 weights).

**Troubleshooting:**

| Error | Cause & Fix |
| :--- | :--- |
| `No merged model found under gs://.../models/` | Step 4 didn't finish, or finished without `--merge-weights`. Confirm with `gcloud storage ls gs://${BUCKET_NAME}/models/**/merged_model_bf16/config.json`. |
| `Model server never became ready` | The merged model directory is incomplete (missing `tokenizer.json` or a `.safetensors` shard). Re-run Step 4. |
| Want a cheaper, scale-to-zero option? | `export DEPLOY_TARGET="cloud-run-gpu"` — Cloud Run with an L4 scales to zero when idle, so you pay only per request-second. Trade-off: multi-second cold starts while the model reloads. |

---

## Step 7 — Smoke Test the Live Endpoint

```bash
./scripts/04_test_endpoint.sh
```

**What this does**: Sends **3 real Indian FSI prompts** to the live endpoint and measures latency:

1. A **UPI failed-transaction TAT** question (should cite `RBI/2019-20/67` and compute ₹100/day compensation).
2. A **KYC Ultimate Beneficial Owner threshold** question (should state **10%**, per the 2023 PMLA amendment — *not* the outdated 25%).
3. A **digital lending KFS / APR + FLDG** audit question (should compute APR including 18% GST and flag the 5% FLDG cap).

**Expected output:**

```text
[TEST 1/3] UPI Failed Transaction TAT & Compensation
  Latency: 1,284 ms
  Response: Per RBI Circular RBI/2019-20/67 (Harmonisation of Turn Around Time),
  a failed UPI transaction where the account is debited but the beneficiary is not
  credited must be auto-reversed within T+1 day. Beyond that, compensation of
  ₹100 per day of delay is payable to the customer...

  ✅ Cited circular:  RBI/2019-20/67
  ✅ Correct penalty: ₹100/day
```

**What to look for** (the script prints this guidance):
- **Correct circular numbers** — a hallucinated circular number is a compliance incident, not a typo.
- **Exact figures** — 10% UBO threshold, 5% FLDG cap, ₹100/day (`₹400` for 4 days past `T+5`), 18% GST.
- **Refusal behaviour** — the model should *never* tell a customer that an STR was filed (PMLA §12A tipping-off prohibition).

### Side-by-Side Comparison: What Untuned `google/gemma-4-12B-it` vs. Your Fine-Tuned Model Outputs on These 3 Prompts

| Test Prompt | ❌ Before Fine-Tuning (`google/gemma-4-12B-it` Base) | ✅ After QLoRA Fine-Tuning (`gemma4-fsi-sovereign-model`) |
| :--- | :--- | :--- |
| **Test 1: UPI P2M Failed Txn (`T+9`, `₹12,500`)** | *"Banks reverse failed UPI transactions in 1–5 days or pay ₹100/day. For 9 days, compensation is ₹900."* *(Misses `RBI/2019-20/67`, misses `UDIR-01`, wrong math `₹900` instead of `₹400`)* | Cites **`RBI/2019-20/67`** + **`UDIR-01`**, applies `T+5` P2M TAT window, and computes $(9 - 5) \times ₹100 = \mathbf{₹400}$ penalty (**`₹12,900`** total credit). |
| **Test 2: High-Risk Pvt Ltd UBO, Re-KYC & STR** | *"UBO threshold is **25%** (FATF standard), Re-KYC is 2 years, and explain transparently if an AML review froze the account."* *(Outdated 25% UBO + criminal PMLA §12A tipping-off!)* | States **`>10%` UBO threshold** (2023 PMLA Rule 9(3) amendment), **`2-year`** Re-KYC, **`7-day`** Rule 8(2) STR timeline, and enforces **PMLA Section 12A anti-tipping-off**. |
| **Test 3: Digital Lending KFS APR & 12% FLDG** | *"APR is ~20.5%. Document the 12% FLDG and LSP nodal escrow routing in the SLA."* *(Approves illegal `>5%` FLDG and illegal LSP escrow pass-through!)* | Computes **`21.03%` KFS APR** (`₹2,500` fee + `₹450` GST) and **blocks disbursement** citing **`DOR.CRE.REC.21/21.07.001/2023-24`** (`5%` FLDG cap) & **Para 3 direct RE-to-borrower flow**. |

**If responses look generic** (no circular numbers, hedging language like "you should consult your bank"), you are almost certainly hitting the **base model, not the fine-tuned one**. Verify with:

```bash
gcloud ai endpoints describe <ENDPOINT_ID> --region="${REGION}" \
  --format="value(deployedModels[0].displayName)"
# Expect: gemma4-fsi-sovereign-model-v1
```

---

## Step 8 — Teardown (MANDATORY)

> [!CAUTION]
> Do not skip this. An idle endpoint costs **`$1.19/hour` (`~$868/month`)**.

### 8a. Dry Run First (Always Safe)

The script is **dry-run by default** and deletes nothing without `--confirm`:

```bash
./scripts/99_cleanup_resources.sh
```

This prints a complete inventory of what it *would* delete, then exits without touching anything.

### 8b. Execute

```bash
# Full teardown — deletes the bucket and your fine-tuned weights too:
./scripts/99_cleanup_resources.sh --confirm

# Recommended — stops all GPU billing but KEEPS your trained model + dataset:
./scripts/99_cleanup_resources.sh --confirm --keep-bucket
```

`--confirm` will prompt you to **type your full project ID** before anything is deleted.

### 8c. Verify Nothing Is Still Billing

```bash
gcloud ai endpoints list --region="${REGION}" --project="${PROJECT_ID}"
gcloud ai custom-jobs list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING"
gcloud run services list --region="${REGION}" --project="${PROJECT_ID}"
```

All three should print **`Listed 0 items.`**

### 8d. Don't Forget the Notebook Runtime

If you used Colab Enterprise or Workbench, the runtime is **billed while RUNNING even if idle**, and the cleanup script deliberately does **not** delete it (you might be mid-session). Stop it manually:

```bash
gcloud colab runtimes list --region="${REGION}" --project="${PROJECT_ID}"
gcloud colab runtimes delete <RUNTIME_ID> --region="${REGION}" --project="${PROJECT_ID}"
```

Or: **Vertex AI Console → Colab Enterprise → Runtimes → Stop**.

For full detail on what each teardown step does and why KMS keys take 24 hours to destroy, see **README Module 11**.

---

## Step 9 — Production Scaling & How to Update / Redeploy the Model (`v1` → `v2`)

Once you move from this single-replica lab to production banking inference at scale:

### 9a. Production Serving Best Practices at Scale
1. **Use vLLM (`PagedAttention` + Continuous Batching + FP8/AWQ)**: Replaces single-stream `model.generate()` (`~7.5 tok/s`) with **vLLM continuous batching (`--max-num-seqs=64`) + `--enable-prefix-caching`**, boosting throughput to **`55–85 tokens/sec`** and cutting Time-to-First-Token (TTFT) to **`< 25 ms`**.
2. **Multi-Zone High Availability & Autoscaling**: Set `--min-replica-count=2 --max-replica-count=10` so Vertex AI spreads replicas across `asia-south1-a` and `asia-south1-b` and autoscales when GPU duty cycle exceeds `70%`.
3. **Private Service Connect (PSC) + VPC-SC + Model Armor**: Expose the endpoint strictly via a private PSC IP inside your bank's VPC-SC perimeter with Google Cloud Model Armor screening prompts and responses.
4. **Dynamic Multi-LoRA on GKE (For 5+ Banking Teams)**: If serving multiple departments (KYC, Lending, Treasury, SEBI, UPI), deploy 1 shared `gemma-4-12B-it` base model on GKE with vLLM `--enable-lora` and hot-swap 130 MB LoRA adapters per request—cutting GPU fleet costs by **70%–85%**.

### 9b. How to Update the Model (`v2`) & Redeploy with Zero Downtime
When a new RBI/SEBI circular is released or you add new training examples:

```bash
# 1. Add new examples, verify zero leakage, and upload updated JSONLs to GCS
python3 data/generate_fsi_dataset.py --samples-per-domain 350 --output-dir data
python3 data/inspect_dataset.py
gcloud storage cp data/*.jsonl data/data_card.json "gs://${BUCKET_NAME}/data/"

# 2. Train v2 (reuses cached container & GCS base mirror — finishes in ~25 mins)
./scripts/02_submit_vertex_finetuning.sh

# 3. Run offline regression gate (must beat v1 + 100% PMLA safety)
python3 evaluation/evaluate_fsi_model.py

# 4. Upload v2 as a new version under the existing parent model in Vertex AI Model Registry
NEW_MODEL_URI=$(gcloud storage ls "gs://${BUCKET_NAME}/models/" | grep '/$' | sort | tail -n 1)merged_model_bf16
PARENT_MODEL_ID=$(gcloud ai models list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="displayName:gemma4-fsi-sovereign-model" --format="value(name)" | head -n 1)

gcloud ai models upload \
  --region="${REGION}" --project="${PROJECT_ID}" \
  --parent-model="${PARENT_MODEL_ID}" \
  --display-name="gemma4-fsi-sovereign-model" \
  --version-aliases="v2,challenger" \
  --container-image-uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/fsi-gemma-repo/gemma-qlora-trainer:v1" \
  --container-env-vars="MODEL_ID=${NEW_MODEL_URI},QUANTIZATION=nf4" \
  --container-command="python3,/app/serve_gemma_vertex.py" \
  --container-ports=8080 --container-predict-route="/predict" --container-health-route="/health"

# 5. Canary deploy v2 with a 90% (v1) / 10% (v2) traffic split on the SAME Endpoint ID
ENDPOINT_ID=$(gcloud ai endpoints list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="displayName:fsi-sovereign-vllm-endpoint" --format="value(name)" | head -n 1)
V1_DEPLOYED_ID=$(gcloud ai endpoints describe "${ENDPOINT_ID}" --region="${REGION}" \
  --project="${PROJECT_ID}" --format="value(deployedModels[0].id)")

gcloud ai endpoints deploy-model "${ENDPOINT_ID}" \
  --region="${REGION}" --project="${PROJECT_ID}" \
  --model="${PARENT_MODEL_ID}@2" \
  --display-name="gemma4-fsi-sovereign-model-v2" \
  --machine-type="g2-standard-12" --accelerator="type=nvidia-l4,count=1" \
  --min-replica-count=1 --max-replica-count=2 \
  --service-account="$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
  --traffic-split="${V1_DEPLOYED_ID}=90,0=10"

# 6. Once verified with ./scripts/04_test_endpoint.sh, cut over 100% traffic to v2 and undeploy v1!
```

See **[README Module 12](../README.md#17-module-12-production-mlops-deploying-at-scale-for-real-world-inference--zero-downtime-model-updates)** for the complete production architecture matrix, cutover commands, and 5-second rollback command.

---

## Appendix A — Multi-Region Configuration, Quota Setup & Accelerator Options

Every script in this repository is parameterized so you can target any Google Cloud region or GPU accelerator family allocated to your project:

### Option 1: Multi-Region Parameterization (`REGION`)

Every script dynamically reads `${REGION}` (defaulting to `asia-south1`):

```bash
export REGION="asia-south1"   # or your project's target region

./scripts/01_setup_sovereign_infra.sh
./scripts/02_submit_vertex_finetuning.sh
```

### Option 2: Configuring the GPU Tier (`NVIDIA L4`, `A100`, or `T4`)

If your project quota is configured for **NVIDIA T4 (`n1-standard-8`)** or you want to train the compact `google/gemma-4-E4B-it` model:

```bash
export MACHINE_TYPE="n1-standard-8"
export ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
export BASE_MODEL_ID="google/gemma-4-E4B-it"   # Compact Gemma 4 E4B model for T4/L4
./scripts/02_submit_vertex_finetuning.sh
```

> [!NOTE]
> **Hardware Recommendation**: NVIDIA L4 (Ada Lovelace) has native `bfloat16` support, whereas T4 (Turing) does not. When using T4, select a compact checkpoint (`google/gemma-4-E4B-it` or `google/gemma-3-4b-it`).

---

## Appendix B — Quick Reference: All Environment Variables

| Variable | Default | Used By |
| :--- | :--- | :--- |
| `PROJECT_ID` | `gcloud config get-value project` | All scripts |
| `REGION` | `asia-south1` | All scripts |
| `HF_TOKEN` | *(none — you must set it)* | Steps 0, 4 |
| `BUCKET_NAME` | `${PROJECT_ID}-fsi-gemma-sovereign-${REGION}` | Steps 3, 4, 6, 8 |
| `KEYRING_NAME` | `fsi-sovereign-kr` | Steps 3, 4, 8 |
| `KEY_NAME` | `gemma-cmek-key` | Steps 3, 4, 8 |
| `AR_REPO_NAME` | `fsi-gemma-repo` | Steps 3, 8 |
| `BASE_MODEL_ID` | `google/gemma-4-12B-it` | Step 4 |
| `BUILD_MODE` | `prebuilt` | Step 4 |
| `MACHINE_TYPE` | `g2-standard-12` | Step 4 |
| `ACCELERATOR_TYPE` | `NVIDIA_L4` | Step 4 |
| `ACCELERATOR_COUNT` | `1` | Step 4 |
| `DEPLOY_TARGET` | `vertex-endpoint` | Step 6 |
| `MODEL_DISPLAY_NAME` | `gemma-indian-fsi-sovereign` | Steps 6, 8 |
| `ENDPOINT_DISPLAY_NAME` | `fsi-sovereign-vllm-endpoint` | Steps 6, 7, 8 |

---

## Where to Go Next

- **[Module 0: Fundamentals for Beginners](01_FUNDAMENTALS_FOR_BEGINNERS.md)** — the theory behind every step above
- **[README Module 8](../README.md#13-module-8-theory--lab-evaluation-driven-development-evals--iterative-hillclimbing-in-fsi)** — Evals & Hillclimbing: how to systematically improve the model
- **[README Module 9](../README.md#14-module-9-deep-dive-encyclopedia-understanding-every-genai-fine-tuning-hyperparameter)** — every hyperparameter explained
- **[README Module 11](../README.md#16-module-11-mandatory-cleanup-teardown--cost-control)** — full cleanup and cost-control reference
