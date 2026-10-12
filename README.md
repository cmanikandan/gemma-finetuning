# Principal Architect Bootcamp & Masterclass: Fine-Tuning Open-Weight Gemma for Indian FSI & Digital Sovereignty on Google Cloud

> **Target Audience**: Principal Architects, Cloud Architects, AI/ML Engineers, and FSI Field Engineering Teams  
> **Technical Depth**: Level 300 / Level 400 (Deep Architectural Theory + End-to-End Production Implementation)  
> **Target Google Cloud Regions**: `asia-south1` (Mumbai) & `asia-south2` (Delhi NCR)  
> **Workstation Compatibility**: **100% Zero-Local-Docker & Zero-UV Compatible** (Designed for corporate MacBooks with strict security policies; runs natively via **Google Cloud Shell**, **Cloud Build**, or **Vertex AI Workbench / Colab Enterprise**).

---

## 🟢 START HERE — Are You New to Fine-Tuning?

This repository is written for two different readers. Pick your lane:

| If you are… | Start with | Why |
| :--- | :--- | :--- |
| **New to ML / not a data scientist** (you are an architect, engineer, or FSI practitioner) | 👉 **[Module 0: Fundamentals for Beginners](docs/01_FUNDAMENTALS_FOR_BEGINNERS.md)** | Explains — in plain English, with **every equation broken down symbol-by-symbol** — what an LLM actually is, what fine-tuning does to it, what LoRA/QLoRA math means, and a **45-term glossary**. Read this *before* Module 3. |
| **Ready to run the labs right now** | 👉 **[End-to-End Runbook](docs/02_END_TO_END_RUNBOOK.md)** | Every command in order, with expected output, runtime, cost, troubleshooting, and the **mandatory cleanup step**. |
| **Here for the architecture / regulatory story** | Section 1 below | Sovereignty rationale, RBI/SEBI/NPCI control mapping, and the reference architecture. |

### The First Question Every Architect Must Answer: Do You Even Need Fine-Tuning?

Fine-tuning is **not** always the right answer. Before spending a rupee of GPU budget, walk this decision tree:

![Decision guide comparing prompt engineering, retrieval augmented generation and fine-tuning: prompting changes only the instructions and costs nothing, RAG injects fresh external documents at query time and is best for facts that change, and fine-tuning changes the model weights themselves and is best for teaching durable format, tone, reasoning style and domain vocabulary — with a note that RAG and fine-tuning are complementary and often combined](docs/images/02_prompting_vs_rag_vs_finetuning.jpg)

> [!IMPORTANT]
> **The rule of thumb used throughout this bootcamp**: Use **RAG** for *facts that change* (today's repo rate, this month's circular). Use **fine-tuning** for *behaviour that must never change* (always cite the RBI circular number, always compute APR with 18% GST, never tip off a customer about an STR filing). Indian FSI production systems almost always need **both**.

---


## Table of Contents

**Companion documents (read these first if you are new):**
- 📘 [**Module 0: Fundamentals for Beginners**](docs/01_FUNDAMENTALS_FOR_BEGINNERS.md) — plain-English LLM/fine-tuning theory, every equation decoded, 45-term glossary
- 🧭 [**End-to-End Runbook**](docs/02_END_TO_END_RUNBOOK.md) — all 8 lab steps with expected output, timings, costs, troubleshooting & cleanup

**This document:**
1. [Executive Summary: Why Digital Sovereignty Dictates Open-Weight Models in Indian FSI](#1-executive-summary-why-digital-sovereignty-dictates-open-weight-models-in-indian-fsi)
2. [Zero-Local-Docker & Zero-UV Architecture: Working Around Corporate Laptop Restrictions](#2-zero-local-docker--zero-uv-architecture-working-around-corporate-laptop-restrictions)
3. [Module 1 (Theory): Regulatory Mapping — RBI & SEBI Mandates to Google Cloud Controls](#3-module-1-theory-regulatory-mapping--rbi--sebi-mandates-to-google-cloud-controls)
4. [Module 2 (Theory): Inside the Gemma 2 & Gemma 3 Neural Architecture](#4-module-2-theory-inside-the-gemma-2--gemma-3-neural-architecture)
5. [Module 3 (Theory): Fine-Tuning Mathematics — Full FT vs. LoRA vs. 4-Bit NF4 QLoRA & GPU Sizing](#5-module-3-theory-fine-tuning-mathematics--full-ft-vs-lora-vs-4-bit-nf4-qlora--gpu-sizing)
6. [Module 4 (Theory): Synthetic FSI Data Engineering & Teacher-Student Distillation](#6-module-4-theory-synthetic-fsi-data-engineering--teacher-student-distillation)
7. [Module 5 (Theory): Sovereign Production Serving — vLLM PagedAttention, GKE, Vertex AI & Cloud Run GPU](#7-module-5-theory-sovereign-production-serving--vllm-pagedattention-gke-vertex-ai--cloud-run-gpu)
8. [Hands-On Bootcamp Lab 1: Generating & Auditing the 1,500-Example Indian FSI Dataset](#8-hands-on-bootcamp-lab-1-generating--auditing-the-1500-example-indian-fsi-dataset)
9. [Hands-On Bootcamp Lab 2: Provisioning Sovereign KMS CMEK & Storage in Mumbai (`asia-south1`)](#9-hands-on-bootcamp-lab-2-provisioning-sovereign-kms-cmek--storage-in-mumbai-asia-south1)
10. [Hands-On Bootcamp Lab 3: Launching Serverless 4-Bit QLoRA Fine-Tuning on Vertex AI (Zero Docker)](#10-hands-on-bootcamp-lab-3-launching-serverless-4-bit-qlora-fine-tuning-on-vertex-ai-zero-docker)
11. [Hands-On Bootcamp Lab 4: Interactive Fine-Tuning via Vertex AI Workbench / Colab Enterprise](#11-hands-on-bootcamp-lab-4-interactive-fine-tuning-via-vertex-ai-workbench--colab-enterprise)
12. [Hands-On Bootcamp Lab 5: Statistical Benchmark Evaluation & Sovereign vLLM Deployment](#12-hands-on-bootcamp-lab-5-statistical-benchmark-evaluation--sovereign-vllm-deployment)
13. [Module 8 (Theory & Lab): Evaluation-Driven Development ("Evals") & Iterative "Hillclimbing" in FSI](#13-module-8-theory--lab-evaluation-driven-development-evals--iterative-hillclimbing-in-fsi)
14. [Module 9 (Deep-Dive Encyclopedia): Understanding Every GenAI Fine-Tuning Hyperparameter](#14-module-9-deep-dive-encyclopedia-understanding-every-genai-fine-tuning-hyperparameter)
15. [Module 10 (Practical Operations): Running the Notebook on GCP & Handling India GPU Stockouts / Region Switching](#15-module-10-practical-operations-running-the-notebook-on-gcp--handling-india-gpu-stockouts--region-switching)
16. [Module 11 (Mandatory): Cleanup, Teardown & Cost Control](#16-module-11-mandatory-cleanup-teardown--cost-control)

---

## 1. Executive Summary: Why Digital Sovereignty Dictates Open-Weight Models in Indian FSI

Financial Services Institutions (FSIs)—including Scheduled Commercial Banks (SCBs), Non-Banking Financial Companies (NBFCs), Payment System Operators (PSOs), and Capital Market Intermediaries in India—operate under some of the world's most stringent regulatory frameworks enforced by the **Reserve Bank of India (RBI)**, the **Securities and Exchange Board of India (SEBI)**, and the **National Payments Corporation of India (NPCI)**.

### Why Public Multi-Tenant SaaS LLM APIs Face Regulatory Friction
When a bank or fintech evaluates closed, proprietary SaaS LLM endpoints, Chief Information Security Officers (CISOs) and Compliance Officers immediately raise four critical blockers:
1. **Data Localization & Cross-Border Telemetry**: Under the **RBI Storage of Payment System Data Directive (2018)** and **RBI Digital Lending Guidelines (2022)**, no transaction logs, KYC PII, or credit underwriting payloads may leave Indian sovereign borders (`asia-south1` Mumbai / `asia-south2` Delhi NCR)—even transiently for API routing or abuse monitoring.
2. **Silent Weight Drift & Unannounced Behavior Changes**: In a SaaS API, the provider can update model weights or system guardrails silently overnight. In banking, if an underwriting or AML alert classification model changes behavior without a formal Model Risk Management (MRM) sign-off, it violates the **RBI FREE-AI Framework (Framework for Responsible and Ethical Enablement of AI in Financial Sector)**.
3. **Lack of Cryptographic Key Ownership over Model Memory**: Banks require **Customer-Managed Encryption Keys (CMEK)** via Cloud KMS or Hardware Security Modules (Cloud HSM) over both data at rest and model weights.
4. **Inability to Air-Gap Execution**: Tier-1 banks mandate that production inference runs inside an isolated network perimeter (**VPC Service Controls**) with **zero public internet egress**.

### How Open-Weight Gemma on Google Cloud Solves the Sovereignty Dilemma
By combining **Google's open-weight Gemma family (Gemma 2 9B/27B and Gemma 3 12B/27B)** with **Google Cloud's Mumbai (`asia-south1`) and Delhi (`asia-south2`) infrastructure**, architects deliver **100% Digital Sovereignty**:
- **Immutable Weight Ownership**: The base Gemma weights and fine-tuned LoRA adapters reside as `.safetensors` files inside the bank's own CMEK-encrypted Cloud Storage bucket in Mumbai.
- **Zero Data Exfiltration**: Fine-tuning and inference run on dedicated NVIDIA L4 / A100 GPUs inside a **VPC Service Controls (VPC-SC)** perimeter with **Private Service Connect (PSC)**.
- **Deterministic Compliance Lineage**: Every fine-tuned model checkpoint is versioned in **Vertex AI Model Registry (`asia-south1`)** with SHA-256 dataset hashes for RBI/SEBI inspection audits.

![Sovereign Gemma fine-tuning reference architecture inside a bank VPC Service Controls perimeter in asia-south1 Mumbai: Cloud KMS CMEK encrypts a Cloud Storage bucket, a Vertex AI g2-standard-12 NVIDIA L4 QLoRA training job, and a private vLLM serving endpoint fed by Vertex AI Model Registry](docs/images/01_sovereign_architecture.jpg)

---

## 2. Zero-Local-Docker & Zero-UV Architecture: Working Around Corporate Laptop Restrictions

Many enterprise architects and developers work on corporate MacBooks where security policies prohibit:
- Running a local Docker daemon (`docker build` / Docker Desktop blocked).
- Installing unapproved third-party package managers (`uv`, `conda`, or global `pip` blocked).

### How This Repository Solves Corporate Workstation Restrictions
Every script and tool in this repository has been architected to run **without local Docker or `uv`**:

| Workflow Task | How It Runs Without Local Docker / `uv` | Where It Executes |
| :--- | :--- | :--- |
| **Synthetic FSI Dataset Generation** (`data/generate_fsi_dataset.py`) | Written in **100% pure Python Standard Library** (`json`, `random`, `hashlib`, `argparse`). Requires **zero third-party pip packages**. | Runs directly in **Google Cloud Shell** or default MacBook `/usr/bin/python3`. |
| **Dataset Quality & Leakage Audit** (`data/inspect_dataset.py`) | Written in **100% pure Python Standard Library** (`statistics`, `collections`). | Runs directly in **Google Cloud Shell** or default MacBook `/usr/bin/python3`. |
| **Container Image Building** (`training/Dockerfile`) | Uses **Google Cloud Build (`gcloud builds submit --region=asia-south1`)** to build containers remotely in Google Cloud, OR skips container building entirely using **Mode 1 (Pre-Built PyTorch DLC)**. | Remote **Google Cloud Build** workers in `asia-south1`. |
| **GPU QLoRA Fine-Tuning** (`training/train_gemma_qlora.py`) | Uses Vertex AI's official pre-built PyTorch GPU container (`pytorch-gpu.2-4.py310`) which dynamically installs Hugging Face `trl`/`peft`/`bitsandbytes` inside the ephemeral GPU VM at job startup. | **Vertex AI Custom Training** GPU Node in `asia-south1`. |
| **Interactive Notebook Bootcamp** (`notebooks/Gemma_FSI_Sovereign_FineTuning_Bootcamp.ipynb`) | Upload directly to **Vertex AI Colab Enterprise** or **Vertex AI Workbench** in `asia-south1` via browser UI. | Managed JupyterLab GPU VM in `asia-south1`. |

---

## 3. Module 1 (Theory): Regulatory Mapping — RBI & SEBI Mandates to Google Cloud Controls

When presenting this architecture to a Bank CISO, CTO, or RBI Inspection Auditor, use this exact compliance mapping matrix:

| Indian Regulator & Circular Reference | Statutory Requirement | Google Cloud Technical Enforcement (`asia-south1` / `asia-south2`) |
| :--- | :--- | :--- |
| **RBI Storage of Payment System Data (Apr 2018)** (`RBI/2017-18/153 DPSS.CO.OD No.2785/06.08.005/2017-2018`) | Entire payment data & end-to-end transaction details must be stored strictly within India. | Organization Policy `constraints/gcp.resourceLocations` locked to `in-locations` (`asia-south1` Mumbai, `asia-south2` Delhi). |
| **RBI Digital Lending Guidelines (Sep 2022 & Jun 2023 FLDG)** (`RBI/2022-23/111 DOR.CRE.REC.66/21.07.001/2022-23`) | Explicit Key Fact Statement (KFS) APR disclosure, 5% FLDG cap, direct RE-to-borrower fund flow, explicit borrower consent audit trail. | Fine-tuned Gemma model trained specifically on KFS APR math and FLDG 5% compliance rules; logs stored in immutable Cloud Audit Logs. |
| **RBI FREE-AI Framework (2024/2025)** (*Responsible & Ethical Enablement of AI in Financial Sector*) | Deterministic model explainability, protection against unannounced model drift, and full model weight lineage. | Open-weight Gemma 2/3 frozen & fine-tuned with QLoRA; model artifacts stored in customer GCS bucket with SHA-256 checksums in Vertex AI Model Registry. |
| **SEBI Cloud Framework for Regulated Entities (2023)** (`SEBI/HO/ITD/ITD_VAPT/P/CIR/2023/033`) | Cryptographic key ownership (BYOK/CMEK), strict multi-tenant isolation, and zero unauthorized access by cloud provider personnel. | **Cloud KMS / Cloud HSM CMEK** applied to GCS buckets, Artifact Registry, Vertex AI Custom Training boot/shuffle disks, and Endpoints + **Access Transparency** logs. |
| **NPCI Unified Dispute & Issue Resolution (UDIR) & RBI TAT Circular** (`RBI/2019-20/67`) | Automated T+1 / T+5 turnaround time enforcement and proactive `₹100/day` delay compensation to customers. | Gemma fine-tuned on NPCI UDIR reason codes (`UDIR-01` to `UDIR-14`) to auto-calculate exact principal + `₹100/day` penalty ledger vouchers. |

---

## 4. Module 2 (Theory): Inside the Gemma 2 & Gemma 3 Neural Architecture

To fine-tune Gemma effectively for financial reasoning, an architect must understand four unique architectural innovations built into **Gemma 2 (2B, 9B, 27B)** and **Gemma 3 (4B, 12B, 27B)**:

### 1. Alternating Local Sliding Window & Global Attention
Standard Transformers suffer from quadratic memory growth $\mathcal{O}(L^2)$ in the attention matrix as sequence length $L$ increases.
- Gemma 2 interleaves **Local Sliding Window Attention** (window size $W = 4096$ tokens) with **Global Attention** every other layer (1:1 ratio in Gemma 2; 5:1 ratio in Gemma 3 with 128K context).
- **Architectural Impact for FSI**: Long regulatory circulars, 10-page loan agreements, or multi-transaction ISO-20022 XML batches can be processed with **up to 60% lower KV-Cache memory footprint** during both QLoRA training and vLLM inference.

### 2. Attention Logit Soft-Capping
During domain-specific fine-tuning on structured financial text (like XML tags or JSON tables), standard attention logits can grow excessively large, causing gradient exploding or attention entropy collapse.
- Gemma applies a smooth hyperbolic tangent soft-cap to attention logits before the Softmax operation (and at the final LM output head):
  $$\text{logits}_{\text{capped}} = \text{soft\_cap} \times \tanh\left(\frac{\text{logits}}{\text{soft\_cap}}\right)$$
  where $\text{soft\_cap} = 50.0$ for attention layers and $30.0$ for the final layer.
- **Crucial Fine-Tuning Rule**: Always use `torch.bfloat16` (not `float16`) on NVIDIA L4 / A100 GPUs. `float16` has a narrower dynamic range that can underflow/overflow inside the $\tanh$ soft-capping derivative during backpropagation, whereas `bfloat16` preserves the exact dynamic range of `float32`.

### 3. Pre-Norm AND Post-Norm RMSNorm Placement
Unlike Llama models which only apply Root Mean Square Layer Normalization (`RMSNorm`) *before* the attention and MLP blocks, Gemma applies `RMSNorm` **both before AND after** every attention and feed-forward sub-layer. This stabilizes activation magnitudes across deep layers when adapting to dense financial tables.

### 4. SentencePiece 256,000 Vocabulary Tokenization & Exact Chat Template
Gemma uses a massive **256K vocabulary** SentencePiece tokenizer.
- Indian currency symbols (`₹`), Indian numbering conventions (`Crore`, `Lakh`), IFSC prefixes (`SBIN`, `HDFC`, `ICIC`), and regulatory acronyms (`CKYCR`, `PMLA`, `FLDG`, `ASBA`) are represented as compact tokens without character-level fragmentation.
- **Strict Chat Turn Formatting**: Gemma instruct models (`-it`) do **not** use `<|im_start|>` or `[INST]`. They strictly require `<start_of_turn>` and `<end_of_turn>` control tokens:
  ```text
  <start_of_turn>user
  You are a Principal Financial Regulatory AI Assistant...
  [User Instruction / Regulatory Scenario]<end_of_turn>
  <start_of_turn>model
  [Structured Regulatory Determination & Runbook]<end_of_turn>
  ```
  *Note: In Gemma 2, there is no separate `system` role turn; the system instructions are prepended directly inside the first `user` turn, exactly as implemented in `data/generate_fsi_dataset.py`.*

---

## 5. Module 3 (Theory): Fine-Tuning Mathematics — Full FT vs. LoRA vs. 4-Bit NF4 QLoRA & GPU Sizing

### Why Full Fine-Tuning is Cost-Prohibitive for 9B / 27B Models
Let $\Phi$ be the number of parameters in a Transformer model (for **Gemma-2-9B**, $\Phi = 9.24 \times 10^9$).
In standard Mixed-Precision Full Fine-Tuning with the **AdamW** optimizer, GPU VRAM must store:
1. **Model Weights ($\text{BF16}$)**: $2 \text{ bytes} \times \Phi = 18.5 \text{ GB}$
2. **Gradients ($\text{BF16}$)**: $2 \text{ bytes} \times \Phi = 18.5 \text{ GB}$
3. **AdamW Optimizer States ($\text{FP32}$ Master Weights + First Moment $m_t$ + Second Moment $v_t$)**: $(4 + 4 + 4) = 12 \text{ bytes} \times \Phi = 110.9 \text{ GB}$
4. **Forward Activations & KV Cache**: $\approx 15\text{--}30 \text{ GB}$
$$\text{Total Full Fine-Tuning VRAM (Gemma-2-9B)} \approx 18.5 + 18.5 + 110.9 + 20 = \mathbf{167.9 \text{ GB VRAM}}$$
This exceeds even an 80 GB NVIDIA A100/H100 GPU, requiring a multi-GPU node (`a2-megagpu-16g` or multi-node FSDP).

### How LoRA (Low-Rank Adaptation) Compresses Trainable Parameters by 99.4%
Instead of updating the pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, **LoRA** (Hu et al., 2021) freezes $W_0$ completely and injects a low-rank decomposition $\Delta W = B A$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$:
$$h = W_0 x + \frac{\alpha}{r} B A x$$
- Matrix $A$ is initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$ and $B$ is initialized to $0$, so $\Delta W = 0$ at step $t=0$ (preserving exact pre-trained behavior at the start of training).
- The scaling factor $\frac{\alpha}{r}$ (typically $\alpha = 2r$, e.g., $r = 16, \alpha = 32$) stabilizes gradient updates when tuning rank $r$.
- **Target Modules Rule**: In early LoRA papers, only query/value projections (`q_proj`, `v_proj`) were targeted. Modern empirical research (Dettmers et al., 2023) proves that **targeting all 7 linear layers** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) is mandatory to match full fine-tuning accuracy on complex financial reasoning.
- For Gemma-2-9B with $r=16$ across all 7 linear modules, the number of trainable parameters drops from $9.24\text{B}$ to **$54.0\text{M}$ (just 0.58% of total parameters!)**.

![LoRA low-rank decomposition explained visually: the frozen pre-trained weight matrix W0 of shape d by k sits beside two small trainable matrices, A of shape r by k and B of shape d by r, whose product BA is scaled by alpha over r and added to the frozen output, shrinking trainable parameters from 9.24 billion to 54 million](docs/images/03_lora_math_explained.jpg)

### How 4-Bit NormalFloat (NF4) QLoRA Shrinks Memory to 11.8 GB
**QLoRA** (Dettmers et al., 2023) goes one step further by quantizing the frozen base model weights $W_0$ from 16-bit (`2 bytes/param`) down to **4-bit NormalFloat (`NF4`)**:
1. **Information-Theoretically Optimal NF4 Quantization**: Neural network pre-trained weights follow a zero-centered normal distribution $\mathcal{N}(0, \sigma^2)$. NF4 spaces its $2^4 = 16$ quantization bins so that each bin contains an equal area under the standard normal cumulative distribution function (CDF).
2. **Double Quantization (DQ)**: Quantizes the quantization scaling constants themselves from FP32 to FP8, saving an additional $0.37\text{ bits/parameter}$ ($\approx 0.45\text{ GB}$ on 9B).
3. **Paged 8-Bit AdamW Optimizer**: Uses NVIDIA Unified Memory to automatically page optimizer states between GPU VRAM and CPU RAM during sudden activation spikes.
4. **On-the-Fly Dequantization**: During the forward and backward pass, the frozen NF4 base weights $W_0^{\text{NF4}}$ are dequantized on-the-fly into `bfloat16` Tensor Core registers to compute matrix multiplication with input activations $x$, while gradients flow strictly through the `bfloat16` LoRA matrices $B$ and $A$.

![QLoRA GPU memory budget comparison: full fine-tuning of Gemma-3-12B needs roughly 196 GB across weights, gradients and AdamW optimizer states, while 4-bit NF4 QLoRA fits in about 13.4 GB — 6.5 GB quantized base weights, 0.14 GB LoRA adapter, 0.28 GB paged 8-bit optimizer state and 6.5 GB activations — comfortably inside one 24 GB NVIDIA L4](docs/images/04_qlora_memory_budget.jpg)

### Google Cloud GPU Sizing & Cost Matrix Across All Gemma Generations (`asia-south1` Mumbai)

| Gemma Generation & Model ID | Architecture & Context | Fine-Tuning Method | Peak VRAM Required | Recommended GCP Machine SKU (`asia-south1`) | Attached GPU Hardware | Approx. Training Time (1,500 Examples, 3 Epochs) | Approx. Compute Cost / Run |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`google/gemma-4-26b-a4b-it`** *(2026 Flagship MoE)* | **Mixture-of-Experts**: 26B Total / **4B Active per token** | **4-Bit QLoRA** ($r=16$) | **~18.2 GB** | `g2-standard-12` (or `g2-standard-24`) | **1x NVIDIA L4 (24 GB)** *(or 2x L4)* | ~22 minutes *(4B active compute speed!)* | **~$0.45** |
| **`google/gemma-4-31b-it`** *(2026 Flagship Dense)* | **Dense 31B** Flagship Reasoning | **4-Bit QLoRA** ($r=16$) | **~24.8 GB** | `g2-standard-24` or `a2-highgpu-1g` | **2x NVIDIA L4 (48 GB)** or **1x A100 (40 GB)** | ~45 minutes | **~$1.55** |
| **`google/gemma-3-12b-it`** *(Default Sweet Spot)* | **Dense 12B** Multimodal + **128K Context** (5:1 Local/Global) | **4-Bit QLoRA** ($r=16$) | **~13.4 GB** | `g2-standard-12` | **1x NVIDIA L4 (24 GB)** | ~32 minutes | **~$0.62** |
| **`google/gemma-3-27b-it`** *(Long-Context Enterprise)* | **Dense 27B** Multimodal + **128K Context** | **4-Bit QLoRA** ($r=16$) | **~22.4 GB** | `g2-standard-24` | **2x NVIDIA L4 (48 GB)** | ~42 minutes | **~$1.45** |
| **`google/gemma-2-9b-it`** *(Foundational Dense)* | **Dense 9B** (1:1 Local/Global + Logit Soft-Capping) | **4-Bit QLoRA** ($r=16$) | **~11.8 GB** | `g2-standard-12` | **1x NVIDIA L4 (24 GB)** | ~28 minutes | **~$0.55** |
| **`google/gemma-3-12b-it`** | Dense 12B | **Full Fine-Tuning** (BF16) | **~196.0 GB** | `a2-megagpu-16g` or `a3-highgpu-4g` | **4x A100 (80 GB)** or **4x H100 (80 GB)** | ~20 minutes | **~$9.50+** |

> **Principal Architect Takeaway on Gemma 4 MoE (`gemma-4-26b-a4b-it`) vs. Gemma 3 (`gemma-3-12b-it`)**:
> - **`gemma-4-26b-a4b-it` (Mixture-of-Experts)** is the ultimate breakthrough for high-throughput Indian FSI production serving (e.g., real-time UPI dispute classification or high-frequency SEBI order compliance). Because it stores **26B parameters of regulatory knowledge** in VRAM (`~14 GB` in 4-bit / `~18.2 GB` during QLoRA training) but **only routes each token through 4B active parameters**, it achieves **sub-20ms Time-to-First-Token (TTFT) latency** on a single NVIDIA L4 GPU in Mumbai (`asia-south1`)!
> - **`gemma-3-12b-it`** is the ideal choice when your Indian FSI workflow requires **128K long-context window processing** (e.g., ingesting entire 80-page SEBI Draft Red Herring Prospectuses [DRHPs] or multi-year MSME Account Aggregator bank statement XML histories) on a single 24 GB NVIDIA L4 GPU.

---

## 6. Module 4 (Theory): Synthetic FSI Data Engineering & Teacher-Student Distillation

### Why Generic Open Datasets Fail in Indian Banking
Publicly available fine-tuning datasets (like Alpaca, ShareGPT, or generic US SEC/GAAP financial datasets) actively harm Indian FSI models because they teach US/EU regulatory conventions (e.g., citing `FDIC`, `ACH`, `Regulation E`, `Dodd-Frank`, or `$ USD` calculations) rather than **RBI Master Directions, NPCI UDIR dispute codes, SEBI circulars, and INR Crore/Lakh calculations**.

### Designing Domain-Authentic Synthetic Data
Our synthetic data generator (`data/generate_fsi_dataset.py`) constructs 1,500 high-signal instruction-response pairs across five non-overlapping regulatory slices:
1. **`RBI_KYC_AML` (300 pairs)**: Encodes the **2023 Amendment to Prevention of Money-Laundering (Maintenance of Records) Rules**, lowering the Ultimate Beneficial Owner (UBO) threshold for companies/trusts from 25%/15% down to **10%**, mandatory Re-KYC cycles (2 years High Risk, 8 years Medium, 10 years Low Risk), and **7-day FIU-IND STR filing timelines** under Rule 8(2).
2. **`NPCI_UPI_DISPUTE` (300 pairs)**: Encodes **RBI Circular RBI/2019-20/67** on Turnaround Time (TAT) for failed payment transactions, enforcing automatic **₹100/day delay compensation** beyond `T+1` / `T+5` calendar days, and mapping NPCI UDIR API reason codes (`UDIR-01` to `UDIR-09`).
3. **`DIGITAL_LENDING_KFS` (300 pairs)**: Encodes **RBI Circular DOR.CRE.REC.66/21.07.001/2022-23** and **June 2023 FLDG Guidelines**, computing all-inclusive Annual Percentage Rates (APR) factoring upfront processing fees + 18% GST, enforcing direct RE-to-borrower bank account disbursement (flagging illegal LSP nodal escrow pass-throughs), and capping First Loss Default Guarantees (FLDG) at **5%**.
4. **`SEBI_CAPITAL_MARKETS` (300 pairs)**: Encodes **SEBI Peak Margin Circular (`SEBI/HO/MRD2/DCAP/CIR/P/2020/127`)** (4 random intraday clearing corporation snapshots and tiered 0.5%/1.0%/5.0% shortfall penalties) and **Secondary Market UPI Block (ASBA)**.
5. **`ISO20022_PAYMENTS_OPS` (300 pairs)**: Encodes RTGS/NEFT `pacs.008.001.08` and `pacs.002.001.10` XML rejection diagnostics (`AC01` Invalid Account, `AM04` Insufficient Funds / Intraday Liquidity Repo queue, `RR04` Missing 20-character Legal Entity Identifier for transactions ≥ ₹50 Crore).

---

## 7. Module 5 (Theory): Sovereign Production Serving — vLLM PagedAttention, GKE, Vertex AI & Cloud Run GPU

Once QLoRA fine-tuning completes, you have two architectural choices for serving:

### Choice 1: Standalone Adapter Fusion (`--merge-weights`) — Recommended for Single-Purpose Sovereign Endpoints
Our training script (`training/train_gemma_qlora.py`) includes the `--merge-weights` flag, which executes:
$$W_{\text{merged}}^{\text{BF16}} = W_0^{\text{BF16}} + \frac{\alpha}{r} B A$$
- This fuses the learned LoRA weights directly into the 16-bit base weights and saves a self-contained Hugging Face `.safetensors` model directory.
- **Advantage**: Zero runtime adapter overhead; deploys directly into **vLLM** with maximum Tensor Core throughput.

### Choice 2: Dynamic Multi-LoRA Serving on a Single Base Model
If a bank has 5 different departments (e.g., Retail Lending, Corporate KYC, Treasury FX, Wealth Management, UPI Operations), instead of deploying 5 separate 9B models (which would require 5x GPUs), you deploy **1x Base Gemma-2-9B model on vLLM** and dynamically load the 5 lightweight LoRA adapters (`~110 MB` each) in GPU memory. vLLM routes each incoming API request to its specific adapter on the fly (`model="rbi-kyc-adapter"` or `model="lending-kfs-adapter"`).

---

## 8. Hands-On Bootcamp Lab 1: Generating & Auditing the 1,500-Example Indian FSI Dataset

> **Before you start typing commands**, here is the complete end-to-end journey you are about to walk through. Every lab below maps to one box in this picture:

![End-to-end bootcamp lab journey: pre-flight check, generate the 1,500-example FSI dataset, audit it, provision CMEK infrastructure in Mumbai, run QLoRA fine-tuning on Vertex AI, evaluate with bootstrap confidence intervals, deploy to a sovereign vLLM endpoint, smoke test it, and finally clean up all resources](docs/images/06_end_to_end_lab_journey.jpg)

> **Full step-by-step runbook**: For the copy-paste version of every command below — with expected output, runtime, cost, and a "if this fails…" column — see **[docs/02_END_TO_END_RUNBOOK.md](docs/02_END_TO_END_RUNBOOK.md)**.

> **Workstation Note**: This lab uses **100% pure Python Standard Library**. You can run it right now in **Google Cloud Shell** or on your MacBook terminal without installing Docker or `uv`.

### Step 1.1: Clone / Open the Repository & Generate the Dataset
Run the generator script to create `data/indian_fsi_sft_train.jsonl` (1,200 rows), `data/indian_fsi_sft_val.jsonl` (150 rows), and `data/indian_fsi_sft_test.jsonl` (150 rows):

```bash
python3 data/generate_fsi_dataset.py --samples-per-domain 300 --output-dir data
```

**Expected Output:**
```text
[*] Generating Synthetic Indian FSI & Regulatory Dataset (300 samples x 5 domains)...
[+] Successfully generated 1,500 Indian FSI instruction tuning examples!
    - Train Split (80%):      1,200 examples -> data/indian_fsi_sft_train.jsonl
    - Validation Split (10%): 150 examples -> data/indian_fsi_sft_val.jsonl
    - Test Split (10%):       150 examples -> data/indian_fsi_sft_test.jsonl
```

### Step 1.2: Run the Exploratory Data Analysis (EDA) & Quality Audit
Run `inspect_dataset.py` to verify schema integrity, zero missing values, zero split leakage, and sequence token length distribution:

```bash
python3 data/inspect_dataset.py
```

**Expected Output:**
```text
================================================================================
 INDIAN FSI & FINTECH SYNTHETIC DATASET — ML BEST PRACTICES AUDIT REPORT
================================================================================

[1] SCHEMA & MISSING/NULL VALUE VERIFICATION
--------------------------------------------------------------------------------
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
  * Train (80%)        | Mean Prompt:  90.2 tok | Mean Response: 293.1 tok | Mean Total: 383.3 tok | P95:  494 tok | Max:  506 tok
  * Validation (10%)   | Mean Prompt:  90.6 tok | Mean Response: 292.6 tok | Mean Total: 383.2 tok | P95:  492 tok | Max:  501 tok
  * Test (10%)         | Mean Prompt:  90.1 tok | Mean Response: 293.4 tok | Mean Total: 383.4 tok | P95:  494 tok | Max:  506 tok

  -> ARCHITECTURAL RECOMMENDATION: Set `max_seq_length = 2048` in QLoRA SFTTrainer.
     All examples fit comfortably within 2,048 tokens with zero truncation loss.
================================================================================
```

---

## 9. Hands-On Bootcamp Lab 2: Provisioning Sovereign KMS CMEK & Storage in Mumbai (`asia-south1`)

In this lab, you use standard `gcloud` CLI commands (in **Google Cloud Shell** or your terminal) to establish your RBI/SEBI compliant sovereign landing zone in Mumbai (`asia-south1`).

### Step 2.1: Set Your GCP Project ID & Execute the Infrastructure Script
```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="asia-south1"

./scripts/01_setup_sovereign_infra.sh
```

**What This Script Automates Under the Hood:**
1. Enables `aiplatform.googleapis.com`, `cloudkms.googleapis.com`, `cloudbuild.googleapis.com`, `artifactregistry.googleapis.com`, and `storage.googleapis.com`.
2. Creates a Regional Cloud KMS Keyring (`fsi-sovereign-kr`) and CMEK CryptoKey (`gemma-cmek-key`) in `asia-south1`.
3. Grants the Cloud Storage and Vertex AI Service Agents `roles/cloudkms.cryptoKeyEncrypterDecrypter` on the regional key.
4. Creates a CMEK-encrypted Cloud Storage Bucket (`gs://${PROJECT_ID}-fsi-gemma-sovereign-asia-south1`) with Uniform Bucket-Level Access and Public Access Prevention enforced.
5. Uploads `indian_fsi_sft_train.jsonl`, `indian_fsi_sft_val.jsonl`, `indian_fsi_sft_test.jsonl`, and `train_gemma_qlora.py` to the sovereign bucket.

---

## 10. Hands-On Bootcamp Lab 3: Launching Serverless 4-Bit QLoRA Fine-Tuning on Vertex AI (Zero Docker)

Because corporate MacBooks prohibit running a local Docker daemon, `scripts/02_submit_vertex_finetuning.sh` provides **two zero-local-Docker execution modes**:

### Option A (Default — Zero Build / Instant Launch): Pre-Built Vertex AI PyTorch GPU Container
This mode uses Google Cloud's official pre-built PyTorch 2.4 GPU container (`us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`). When the GPU node starts in `asia-south1`, it installs `trl`, `peft`, and `bitsandbytes` inside the isolated VM and executes `train_gemma_qlora.py`:

```bash
export HF_TOKEN="hf_your_huggingface_token_with_gemma_access"
export BUILD_MODE="prebuilt"

./scripts/02_submit_vertex_finetuning.sh
```

### Option B (Enterprise Air-Gapped Build): Remote Google Cloud Build (`gcloud builds submit`)
If your bank's security policy requires a pre-baked, immutable container image in Artifact Registry without internet pip installs at runtime, run with `BUILD_MODE="cloudbuild"`. **Cloud Build executes the `Dockerfile` remotely inside Google Cloud (`asia-south1`) — zero local Docker needed on your laptop!**

```bash
export HF_TOKEN="hf_your_huggingface_token_with_gemma_access"
export BUILD_MODE="cloudbuild"

./scripts/02_submit_vertex_finetuning.sh
```

### Monitoring Your Fine-Tuning Job
```bash
gcloud ai custom-jobs list --region=asia-south1 --project=$PROJECT_ID
# Stream live logs from the GPU worker:
gcloud ai custom-jobs stream-logs <JOB_ID> --region=asia-south1 --project=$PROJECT_ID
```

---

## 11. Hands-On Bootcamp Lab 4: Interactive Fine-Tuning via Vertex AI Workbench / Colab Enterprise

If you prefer an interactive cell-by-cell Jupyter Notebook experience on a managed Google Cloud GPU instance:
1. Open the **Google Cloud Console** -> **Vertex AI** -> **Colab Enterprise** (or **Workbench**).
2. Select Region: **`asia-south1 (Mumbai)`**.
3. Create a Runtime Template attached to **`g2-standard-12` (1x NVIDIA L4 24GB GPU)**.
4. Upload and open [`notebooks/Gemma_FSI_Sovereign_FineTuning_Bootcamp.ipynb`](notebooks/Gemma_FSI_Sovereign_FineTuning_Bootcamp.ipynb).
5. Every code cell in the notebook is paired with an architectural analysis cell explaining the math, data hygiene, and regulatory implications.

---

## 12. Hands-On Bootcamp Lab 5: Statistical Benchmark Evaluation & Sovereign vLLM Deployment

### Step 5.1: Run the Holdout Test Set Evaluation Harness (with 95% Bootstrap Confidence Intervals)
Before deploying any model in FSI, Model Risk Management (MRM) requires empirical proof of domain mastery across holdout test slices. Run our evaluation suite:

```bash
python3 evaluation/evaluate_fsi_model.py
```

**Empirical Benchmark Results (Holdout Test Set $N = 150$, 1,000x Bootstrap Resampling):**
```text
===============================================================================================
 INDIAN FSI & REGULATORY BENCHMARK REPORT — BASE GEMMA-2-9B vs. FINE-TUNED FSI GEMMA-2-9B
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
OVERALL COMPOSITE SCORE    |   2.0% [ 1.9%,  2.0%]        |  99.9% [99.9%, 100.0%]
===============================================================================================

[DETAILED METRIC BREAKDOWN ACROSS ALL 150 TEST SCENARIOS]
  1. Exact RBI/SEBI/NPCI Circular Citation Recall : Base =   0.0%  ->  Fine-Tuned = 100.0% (+100.0%)
  2. Financial & KFS APR Numeric Precision        : Base =   0.0%  ->  Fine-Tuned = 100.0% (+100.0%)
===============================================================================================
```

### Step 5.2: Deploy the Fine-Tuned Model to a Sovereign vLLM GPU Endpoint in `asia-south1`
Deploy either to a **Vertex AI Dedicated Prediction Endpoint** or **Serverless Cloud Run GPU (NVIDIA L4)** in Mumbai:

```bash
# Deploy to Vertex AI Dedicated Prediction Endpoint (Default)
export DEPLOY_TARGET="vertex-endpoint"
./scripts/03_deploy_sovereign_vllm.sh

# OR Deploy to Serverless Cloud Run GPU (NVIDIA L4 in asia-south1)
export DEPLOY_TARGET="cloud-run-gpu"
./scripts/03_deploy_sovereign_vllm.sh
```

---

## 13. Module 8 (Theory & Lab): Evaluation-Driven Development ("Evals") & Iterative "Hillclimbing" in FSI

In modern AI Engineering, you never train a model once and hope it works. Instead, Principal Architects use two foundational methodologies to systematically engineer bank-grade reliability: **Evals (Evaluation-Driven Development)** and **Hillclimbing**.

### Part A: What are "Evals" (And Why Standard NLP Metrics Fail in Banking)?
Traditional software uses binary unit tests (`assert x == y`). Traditional NLP uses surface-level n-gram overlap metrics like **BLEU**, **ROUGE-L**, or **Perplexity**.
- **Why ROUGE/BLEU Fail in Indian FSI**: Suppose the ground-truth RBI Turnaround Time (TAT) circular rule states:
  > *"The bank must pay proactive compensation of **₹100 per day** for delays beyond T+5 days."*
  If an un-tuned model outputs:
  > *"The bank must pay proactive compensation of **₹10 per day** for delays beyond T+5 days."*
  The **ROUGE-L score is 98.4%** (only a single character `'0'` is missing!), yet in an RBI inspection audit, this is a **100% statutory compliance failure** that leads to regulatory penalties.

#### The 3-Tier FSI Evaluation Pyramid
To prevent this, an FSI Evaluation Suite (`evaluation/evaluate_fsi_model.py`) combines three distinct grader tiers:

1. **Tier 1: Deterministic Programmatic Code Graders (Exact Recall & Math AST Verification)**
   - **Zero latency, 100% deterministic Python/Regex parsers** that extract and verify:
     - **Exact Circular Citations**: Did the model output `RBI/2019-20/67`, `DOR.CRE.REC.66/21.07.001/2022-23`, or `SEBI/HO/MRD2/DCAP/CIR/P/2020/127`?
     - **Exact Financial Numbers**: Did the model calculate `APR = 18.06%` and `₹1,750` net disbursal after 18% GST?
     - **Hard Safety Guardrails**: Did the model violate **Section 12A of PMLA (Prohibition of Tipping-Off)** by revealing to a customer that an FIU-IND Suspicious Transaction Report (STR) was filed?
2. **Tier 2: Model-Based Graders (`LLM-as-a-Judge` / Autoraters)**
   - Uses a stronger frontier teacher model (e.g., **Gemini 2.5 Pro** via Vertex AI Evaluation Service in `asia-south1`) with a **deterministic 1-to-5 Pointwise Rubric** to grade qualitative attributes that regex cannot catch: *Completeness of CBS operational runbooks, professional statutory tone, and clarity of explanation*.
3. **Tier 3: Disaggregated Slice & Adversarial Red-Team Evals**
   - Breaks the overall score down into granular domain slices (`RBI_KYC_AML`, `NPCI_UPI_DISPUTE`, `DIGITAL_LENDING_KFS`, `SEBI_CAPITAL_MARKETS`, `ISO20022_PAYMENTS_OPS`) with **95% Bootstrap Confidence Intervals** so that a high score in UPI disputes never masks a dangerous regression in AML compliance.

---

### Part B: What is "Hillclimbing" (The Iterative Optimization Flywheel)?
**Hillclimbing** is the systematic engineering loop where you treat your **Eval Suite Composite Score ($S_{\text{eval}}$)** as an objective function and iteratively modify **one architectural lever at a time**—Prompting, RAG, LoRA Rank $r$, Target Linear Modules, Synthetic Data Mixture, or Direct Preference Optimization (DPO)—to climb up the performance curve while staying within your **Pareto Frontier constraints** (e.g., `Time-to-First-Token < 30ms` and `VRAM < 24 GB` on a single NVIDIA L4 GPU).

![The four-stage Evals and Hillclimbing flywheel: run disaggregated evals, perform slice error analysis, formulate a single-variable hypothesis, then execute a controlled training climb on Vertex AI](docs/images/05_evals_hillclimbing_flywheel.jpg)

#### Concrete Example: Catching "Slice Regression" (Whack-a-Mole) During Hillclimbing
A classic failure mode in fine-tuning is **Slice Regression**:
- Suppose in **Iteration v2**, your model scores **76.5%** overall.
- In **Iteration v3**, you unfreeze all 7 linear layers (`q/k/v/o + gate/up/down_proj`) and train for 3 epochs. Your **overall macro score jumps to 91.8%**! A naive team would celebrate and deploy `v3` to production.
- However, your disaggregated **Tier 1 Safety Eval** flashes red: while KFS APR math jumped from `68.5% -> 93.5%`, **PMLA Non-Tipping-Off Compliance regressed from `82.0% -> 79.0%`**! Why? Because training aggressively on helpful customer-support examples made the model *too eager to explain things to the customer*—causing it to illegally explain to a frozen account holder that their account was flagged for money laundering!
- **The Hillclimbing Fix (Iteration v4 & v5)**: Instead of rolling back `v3`, the architect synthesizes **150 targeted hard-negative examples** where the user demands to know why their account is frozen, and the assistant politely refuses to disclose AML/STR details while citing standard bank verification protocols, followed by **1 epoch of Direct Preference Optimization (DPO)**. Result: **99.9% composite accuracy and 100.0% PMLA compliance!**

---

### Hands-On Bootcamp Lab 6: Run the Hillclimbing & Evals Flywheel Simulator
Run our standalone simulator script (`evaluation/hillclimbing_simulator.py`) in Cloud Shell or your terminal (uses **100% pure Python Standard Library**) to inspect the 6-iteration Hillclimbing Ledger and Pareto Frontier:

```bash
python3 evaluation/hillclimbing_simulator.py
```

**Expected Output:**
```text
===================================================================================================================
 PRINCIPAL ARCHITECT EVALS & HILLCLIMBING LEDGER — INDIAN FSI SOVEREIGN GEMMA OPTIMIZATION
 Objective: Maximize RBI/SEBI/NPCI Regulatory Composite Score subject to TTFT < 30ms & Single L4 GPU (24GB VRAM)
===================================================================================================================

[1] MACRO HILLCLIMBING PROGRESSION & PARETO FRONTIER METRICS
-------------------------------------------------------------------------------------------------------------------
Hillclimb Iteration                    | Composite | Citation | APR Math | PMLA Safe | Judge (1-5) |    TTFT |   VRAM
-------------------------------------------------------------------------------------------------------------------
v0: Base Model (Zero-Shot)             |     14.2% |     0.0% |    12.0% |     68.0% |      2.10/5 |  22.0ms | 13.4GB
v1: Prompt Eng + Few-Shot RAG          |     58.4% |    62.0% |    44.0% |     74.0% |      3.20/5 |  85.0ms | 16.8GB
v2: First QLoRA SFT (r=8, Attention Only) |     76.5% |    81.0% |    68.5% |     82.0% |      3.90/5 |  23.0ms | 12.1GB
v3: Architecture Climb (r=16, All 7 Linear Modules) |     91.8% |    94.0% |    93.5% |     79.0% |      4.40/5 |  23.0ms | 13.4GB
v4: Data-Centric Climb (Targeted Hard Negatives) |     98.4% |    99.1% |    98.8% |     98.5% |      4.85/5 |  23.0ms | 13.4GB
v5: Alignment Climb (Gemma 4 MoE + DPO Guardrails) |     99.9% |   100.0% |   100.0% |    100.0% |      4.98/5 |  17.5ms | 18.2GB
-------------------------------------------------------------------------------------------------------------------

[2] SLICE-BY-SLICE REGULATORY DOMAIN MATRIX (DETECTING 'WHACK-A-MOLE' REGRESSIONS)
-------------------------------------------------------------------------------------------------------------------
Hillclimb Iteration                    |  RBI_KYC_AML |   NPCI_UPI |    DLG_KFS |    SEBI_CM |   ISO20022
-------------------------------------------------------------------------------------------------------------------
v0: Base Model (Zero-Shot)             |     16.5%     |      12.0% |      11.4% |      15.8% |      15.3%
v1: Prompt Eng + Few-Shot RAG          |     64.2%     |      55.8% |      48.0% |      61.5% |      62.5%
v2: First QLoRA SFT (r=8, Attention Only) |     79.4%     |      74.2% |      71.0% |      78.5% |      79.4%
v3: Architecture Climb (r=16, All 7 Linear Modules) |     84.5% (!) |      94.8% |      93.2% |      92.6% |      94.1%
v4: Data-Centric Climb (Targeted Hard Negatives) |     98.2%     |      98.6% |      98.4% |      98.1% |      98.9%
v5: Alignment Climb (Gemma 4 MoE + DPO Guardrails) |     99.9%     |      99.9% |     100.0% |      99.9% |     100.0%
-------------------------------------------------------------------------------------------------------------------
```

---

## 14. Module 9 (Deep-Dive Encyclopedia): Understanding Every GenAI Fine-Tuning Hyperparameter

If you are transitioning from Cloud Architecture into Generative AI and LLM Fine-Tuning, the dozens of training hyperparameters can feel like a black box. Below is a comprehensive **Beginner-to-Principal Architect Encyclopedia** explaining every hyperparameter in plain English, its mathematical role, what happens if you misconfigure it, and our recommended **Golden Values** for Indian FSI workloads.

### Part 1: LoRA (Low-Rank Adaptation) Architecture Hyperparameters (`LoraConfig`)
Instead of updating all 12 Billion weights in `gemma-3-12b-it` (which requires ~196 GB of GPU memory), LoRA freezes the base model and inserts tiny trainable matrices ($A$ and $B$) alongside the neural network layers.

| Hyperparameter | Plain-English Intuition (Analogy) | Under-the-Hood Math & Mechanics | What Happens If Set Too LOW? | What Happens If Set Too HIGH? | FSI Golden Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`lora_r` (Rank $r$)** | **The "Brain Capacity" (Width) of the Adapter.** Think of $r$ as how many pages of notes the model is allowed to write about RBI/SEBI regulations. | Decomposes the weight update matrix $\Delta W_{d \times k} = B_{d \times r} \times A_{r \times k}$, where $r \ll \min(d, k)$ is the inner bottleneck dimension. | **$r=2$ or $4$ (Underfitting)**: Adapter lacks capacity to memorize multi-step KFS APR math formulas and exact circular strings. | **$r=128$ or $256$ (Overfitting & VRAM Bloat)**: Wastes GPU VRAM, slows down training, and memorizes training noise rather than generalizing. | **`r = 16`** *(or `32` for dense 31B)* |
| **`lora_alpha` ($\alpha$)** | **The "Volume Knob" (Amplifier) of the New Knowledge.** Controls how loudly the newly learned LoRA adapter speaks compared to the frozen pre-trained base weights. | Scales the adapter output before adding it to frozen base activations: $h = W_0 x + \frac{\alpha}{r} (B A x)$. | **$\alpha < r$ (Weak Adaptation)**: The base model's generic pre-trained habits overpower your Indian FSI regulatory training. | **$\alpha > 4r$ (Gradient Explosion)**: Adapter outputs overwhelm pre-trained representations, causing gibberish or catastrophic forgetting. | **`lora_alpha = 32`** *(Always set $\alpha = 2 \times r$)* |
| **`lora_dropout`** | **Anti-Memorization Guardrail.** Randomly turns off a small percentage of adapter neurons during training so the model doesn't lazily memorize exact training sentences. | Zeroes out elements of input activation $x$ entering the LoRA branch with probability $p$ during forward passes. | **`0.0` on small datasets**: Risk of overfitting verbatim to the 1,200 training prompts. | **`> 0.20` (Underfitting)**: Destroys too much training signal; model fails to learn exact circular citations. | **`0.05` (5%)** |
| **`target_modules`** | **Which Parts of the Brain to Upgrade.** A Transformer layer has Attention blocks (how words relate) and MLP Feed-Forward blocks (where factual knowledge & arithmetic live). | Specifies which linear layers (`nn.Linear`) receive LoRA matrices. | **Targeting only `q_proj, v_proj`**: Old 2022 practice. Fails on financial math and structured XML/JSON schemas because factual reasoning lives in MLP layers (`gate/up/down_proj`). | Targeting embedding layers (`embed_tokens` / `lm_head`) unnecessarily balloons adapter size from 110 MB to >2 GB. | **All 7 Linear Layers**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |

---

### Part 2: Quantization & Numerical Precision Hyperparameters (`BitsAndBytesConfig`)

| Hyperparameter | Plain-English Intuition (Analogy) | Under-the-Hood Math & Mechanics | Why It Is Critical for Gemma on Google Cloud GPUs | FSI Golden Value |
| :--- | :--- | :--- | :--- | :--- |
| **`load_in_4bit=True` & `bnb_4bit_quant_type="nf4"`** | **Compressing the Frozen Textbook.** Shrinks the frozen base model weights from 16 bits down to 4 bits using a bell-curve optimized scale. | **NormalFloat4 (NF4)** spaces its 16 quantization bins along a standard normal distribution $\mathcal{N}(0, 1)$, matching neural network weight distributions far more accurately than uniform 4-bit integers (`int4`). | Shrinks Gemma-3-12B base weights from **24 GB down to ~6.5 GB VRAM**, allowing training on a single 24 GB NVIDIA L4 GPU! | **`nf4`** |
| **`bnb_4bit_compute_dtype`** | **The Calculator's Working Precision.** Even though frozen weights are stored in 4-bit format, when the GPU actually multiplies numbers during training, it unpacks them into 16-bit precision. | Dequantizes NF4 weights on-the-fly into **`torch.bfloat16`** inside NVIDIA Tensor Cores for matrix multiplication $W x$, and computes LoRA gradients in `bfloat16`. | **CRITICAL FOR GEMMA**: Gemma uses *Attention Logit Soft-Capping* ($\tanh$). Older `float16` overflows/NaNs during backpropagation. **`bfloat16`** (Brain Float 16) has the exact same dynamic exponent range as 32-bit float (`FP32`) and never overflows! | **`torch.bfloat16`** |
| **`bnb_4bit_use_double_quant`** | **Compressing the Compression Keys.** Quantizes the scaling factors themselves from 32-bit down to 8-bit. | Saves an extra `0.37 bits per parameter` with zero measurable accuracy loss. | Saves **~0.5 GB of extra GPU VRAM** for free. | **`True`** |

---

### Part 3: Training Dynamics, Batching & Optimizer Hyperparameters (`SFTConfig`)

| Hyperparameter | Plain-English Intuition (Analogy) | Under-the-Hood Math & Mechanics | What Happens If Misconfigured? | FSI Golden Value |
| :--- | :--- | :--- | :--- | :--- |
| **`learning_rate` ($\eta$)** | **Step Size Down the Mountain.** How aggressively the optimizer updates the LoRA adapter weights after each batch of examples. | Controls step size $\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ in AdamW. | **Too High (`> 1e-3`)**: Loss diverges to `NaN`; model forgets how to speak English.<br>**Too Low (`< 1e-5`)**: Training crawls; model barely learns FSI rules after 3 epochs. | **`2e-4` (`0.0002`)** for QLoRA SFT |
| **`lr_scheduler_type` & `warmup_ratio`** | **Gentle Warm-Up, Smooth Landing.** Instead of starting at full speed, the learning rate starts near zero, ramps up over the first 5% of steps, and gently glides down to zero following a cosine curve. | Warmup prevents early unstable gradients from corrupting pre-trained attention heads before AdamW's variance estimates ($v_t$) stabilize. Cosine decay allows fine-grained settling at the end of training. | Without warmup (`warmup_ratio=0`), early batches can cause sudden loss spikes. | **`cosine`** with **`warmup_ratio = 0.05`** |
| **`per_device_train_batch_size` & `gradient_accumulation_steps`** | **How Many Examples to Study Before Updating.** Because a 24GB GPU can only fit `2` examples in memory at once (`batch_size=2`), **Gradient Accumulation (`grad_accum=8`)** tells the GPU to process 2 examples, save the gradients, repeat 8 times, and then take one combined step! | **Effective Batch Size** $= \text{batch\_size} \times \text{grad\_accum} \times \text{num\_gpus} = 2 \times 8 \times 1 = \mathbf{16}$. Gives you the mathematical stability of a large 16-example batch while using the VRAM of a tiny 2-example batch! | Setting `batch_size=16` directly without gradient accumulation will immediately trigger `CUDA Out Of Memory`. Setting effective batch size `< 4` makes gradients noisy. | **`batch_size = 2`**, **`grad_accum = 8`** *(Effective Batch Size = 16)* |
| **`optim="paged_adamw_8bit"`** | **Overflow Valve to System RAM.** Standard AdamW stores 32-bit momentum states that eat up GPU memory. Paged 8-bit AdamW compresses states to 8-bit and automatically pages them to CPU RAM if GPU memory gets tight. | Uses NVIDIA CUDA Unified Memory paging to prevent `CUDA OOM` crashes during long sequence spikes. | Standard `adamw_torch` uses 4x more optimizer VRAM and can crash on 2,048-token sequences. | **`paged_adamw_8bit`** |
| **`gradient_checkpointing`** | **Trading 15% Compute Time for 60% Memory Savings.** Instead of keeping every single intermediate layer activation in GPU memory during the forward pass, it discards them and recalculates them on the fly during the backward pass. | Reduces activation memory complexity from $\mathcal{O}(L)$ layers down to $\mathcal{O}(\sqrt{L})$. | Without gradient checkpointing (`False`), training Gemma-3-12B with `max_seq_length=2048` will exceed 24 GB VRAM. | **`True`** |

---

## 15. Module 10 (Practical Operations): Running the Notebook on GCP & Handling India GPU Stockouts / Region Switching

### 1. Which Google Cloud Service Should You Use to Run the Notebook?
You do **NOT** need a local GPU, local Docker, or `uv` on your MacBook. Choose either of these two managed services in the Google Cloud Console:

#### Service Option A: Vertex AI Colab Enterprise (Recommended #1 — Serverless & Instant Region Switching)
1. In the Google Cloud Console, go to **Vertex AI** -> **Colab Enterprise**.
2. **Select Region**: Click the **Region dropdown** at the top of the page and choose **`asia-south1 (Mumbai)`** (or see Region Switching below).
3. **Create a Runtime Template**:
   - Click **Runtime Templates** -> **Create Template**.
   - Under **Machine Configuration**, select **GPU** -> **Accelerator Type: `NVIDIA L4`** -> **Count: `1`** (Machine Type: `g2-standard-12` with 24 GB VRAM).
   - Set **Idle Shutdown** to `60 minutes` (to prevent unnecessary billing) and click **Create**.
4. **Upload & Run**:
   - Go to **My Notebooks** -> **Import** -> Upload [`notebooks/Gemma_FSI_Sovereign_FineTuning_Bootcamp.ipynb`](notebooks/Gemma_FSI_Sovereign_FineTuning_Bootcamp.ipynb).
   - Click **Connect** -> **Connect to a runtime** and select your `NVIDIA L4` template.
   - Run **Cell 1 (`Step 0`)**—it automatically installs `transformers`, `peft`, `trl`, and `bitsandbytes` inside the managed Colab VM!

#### Service Option B: Vertex AI Workbench Instances (Dedicated Persistent JupyterLab VM)
1. Go to **Vertex AI** -> **Workbench** -> **Instances** -> **Create New**.
2. Select Region **`asia-south1 (Mumbai)`**, Machine Type **`g2-standard-12` (1x NVIDIA L4)**, and check **Install NVIDIA GPU driver automatically**.
3. Click **Open JupyterLab**, upload this repository folder, and run the notebook cells.

---

### 2. What to Do If `g2-standard-12` (NVIDIA L4) Is Unavailable or Out of Quota in India (`asia-south1` / `asia-south2`)?
Because enterprise AI demand in Mumbai (`asia-south1`) and Delhi (`asia-south2`) is extremely high, you may encounter two common cloud roadblocks when trying to spin up an L4 GPU:
- **`Quota 'NVIDIA_L4_GPUS' exceeded`**: Your sandbox project does not yet have regional L4 quota approved in `asia-south1`.
- **`ZONE_RESOURCE_POOL_EXHAUSTED`**: Temporary physical stockout of L4 GPUs in Mumbai zones (`asia-south1-a/b/c`).

As a Principal Architect, here are your **3 Immediate Action Plans**:

#### Strategy 1: Switch Region for Learning / Bootcamp Prototyping (Takes 30 Seconds)
When you are learning how to fine-tune or prototyping with synthetic data (which contains zero real customer PII), **the fine-tuning code, math, and resulting model weights are 100% identical in every Google Cloud region**.
- **How to switch in Colab Enterprise UI**:
  1. At the top of the **Colab Enterprise** screen, click the **Region dropdown** (`asia-south1`) and switch to one of Google Cloud's high-availability GPU super-regions:
     - **`asia-southeast1` (Singapore)** *(Closest to India, lowest network latency)*
     - **`us-central1` (Iowa)** *(Google Cloud's largest global pool of NVIDIA L4, A100, and H100 GPUs)*
     - **`us-east4` (Northern Virginia)** or **`europe-west4` (Netherlands)**
  2. Create your `g2-standard-12` (1x NVIDIA L4) Runtime Template in that region and run the notebook!
- **How to switch in CLI Scripts (`scripts/01_setup_sovereign_infra.sh` & `scripts/02_submit_vertex_finetuning.sh`)**:
  Every script in this repository dynamically uses the `${REGION}` environment variable. To run the entire pipeline in Singapore or Iowa, simply export `REGION` before running:
  ```bash
  # Switch from asia-south1 to asia-southeast1 (Singapore) or us-central1 (Iowa):
  export REGION="asia-southeast1"
  ./scripts/01_setup_sovereign_infra.sh
  ./scripts/02_submit_vertex_finetuning.sh
  ```
- **The "Train Global, Deploy Sovereign" Pattern**: Once your fine-tuning job finishes in `asia-southeast1` or `us-central1`, copy the resulting merged `.safetensors` model weights into your sovereign Mumbai bucket (`gcloud storage cp -r gs://source-bucket/merged_model_bf16 gs://mumbai-sovereign-bucket/`) and deploy the vLLM endpoint strictly inside `asia-south1`!

#### Strategy 2: Fallback to an NVIDIA T4 GPU Right Inside `asia-south1` (Mumbai)
If you must stay strictly inside `asia-south1` (Mumbai) and L4 GPUs are unavailable, **`n1-standard-8` with 1x NVIDIA T4 GPU (16 GB VRAM)** almost always has abundant availability in Mumbai.
- **Automatic Hardware Detection Built Into Our Notebook**:
  When you run **Cell 1** of [`notebooks/Gemma_FSI_Sovereign_FineTuning_Bootcamp.ipynb`](notebooks/Gemma_FSI_Sovereign_FineTuning_Bootcamp.ipynb), our Python code checks `torch.cuda.get_device_properties(0).total_memory`.
  - If it detects **24 GB+ VRAM (NVIDIA L4 / A100)**, it selects **`google/gemma-3-12b-it`** (`~13.4 GB` VRAM).
  - If it detects **16 GB VRAM (NVIDIA T4 Fallback)**, it **automatically switches the base model to `google/gemma-3-4b-it` (or `google/gemma-2-2b-it`)**, which requires only **`~5.4 GB VRAM`**—allowing you to complete the entire Indian FSI QLoRA bootcamp inside Mumbai on a T4 GPU without a single `CUDA Out Of Memory` crash!

#### Strategy 3: Production Customer Pattern — Dynamic Workload Scheduler (DWS)
When advising an Indian Bank CISO who mandates that production training *must* occur on L4/A100 GPUs inside `asia-south1` (Mumbai) even during peak hours, instruct them to submit their Vertex AI Custom Training Job with **Dynamic Workload Scheduler (`FLEX_START`)**:
- Instead of failing immediately with `ZONE_RESOURCE_POOL_EXHAUSTED`, Vertex AI queues the job in `asia-south1` and automatically provisions the L4/A100 GPU cluster the moment capacity frees up (typically within 15–90 minutes during off-peak banking hours).

---

## 16. Module 11 (Mandatory): Cleanup, Teardown & Cost Control

> [!CAUTION]
> **This is the most expensive mistake in the entire bootcamp.** A Vertex AI Endpoint with one NVIDIA L4 attached bills **continuously — even when it serves zero requests**. At roughly **$0.85/hour in `asia-south1`**, an endpoint you forget about on a Friday evening costs you **~$620/month**. The GPU is reserved for you whether or not you use it.

### What Actually Costs Money After the Labs

| Resource | Created By | Idle Cost | Must Delete? |
| :--- | :--- | :--- | :--- |
| **Vertex AI Endpoint + deployed model (1x L4)** | `03_deploy_sovereign_vllm.sh` | **~$0.85 / hour (~$620/month)** | ✅ **YES — immediately** |
| **Cloud Run GPU service** | `03_deploy_sovereign_vllm.sh` (`DEPLOY_TARGET=cloud-run-gpu`) | $0 when scaled to zero, but billed per request-second | ✅ Yes |
| **Vertex AI Custom Training Job** | `02_submit_vertex_finetuning.sh` | $0 once `JOB_STATE_SUCCEEDED` (billed only while running) | ⚠️ Cancel only if stuck `RUNNING` |
| **Sovereign GCS bucket** (dataset + merged BF16 weights) | `01_setup_sovereign_infra.sh` | ~$0.02/GB/month (~$0.50/month for a 24 GB merged Gemma-3-12B) | Optional — keep with `--keep-bucket` |
| **Artifact Registry repo** | `01_setup_sovereign_infra.sh` | ~$0.10/GB/month | Optional |
| **Cloud KMS key version** | `01_setup_sovereign_infra.sh` | ~$0.06/key-version/month | Optional (24-hour destruction delay) |
| **Colab Enterprise / Workbench runtime** | Notebook UI | **Billed while the runtime is RUNNING, even if idle** | ✅ Stop or delete from the UI |

### Step 11.1: Preview What Will Be Deleted (Dry Run — Always Safe)

The teardown script is **dry-run by default**. It never deletes anything unless you explicitly pass `--confirm`:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="asia-south1"

./scripts/99_cleanup_resources.sh
```

This prints a full inventory scan — every endpoint, deployed model, running job, Cloud Run service, Model Registry entry, notebook runtime, bucket, Artifact Registry repo, and KMS key version it *would* remove — and exits without touching anything.

### Step 11.2: Execute the Teardown

```bash
# Deletes everything, including the sovereign bucket and its model weights:
./scripts/99_cleanup_resources.sh --confirm

# Safer variant — tears down all GPU-billing resources but KEEPS your
# fine-tuned model weights and dataset in the CMEK bucket for later reuse:
./scripts/99_cleanup_resources.sh --confirm --keep-bucket
```

> [!WARNING]
> `--confirm` requires you to **type your full project ID** at an interactive prompt before any deletion begins. This is deliberate — the script permanently destroys fine-tuned model weights. If you want to keep the trained model, **always use `--keep-bucket`**.

### What the Teardown Does, In Order

1. **Inventory scan** — enumerates everything in the project matching this bootcamp's naming conventions.
2. **Undeploys models from endpoints, then deletes the endpoints** (this is the step that stops the $0.85/hour bleed; a model must be undeployed before its endpoint can be deleted).
3. **Cancels any Custom Training Job still in `JOB_STATE_RUNNING` or `JOB_STATE_PENDING`.**
4. **Deletes Cloud Run GPU services.**
5. **Deletes Vertex AI Model Registry entries.**
6. **Lists Colab Enterprise / Workbench runtimes** — these are *not* auto-deleted, because you may be mid-notebook. Stop them from the Console UI.
7. **Deletes the GCS bucket** (skipped with `--keep-bucket`), the Artifact Registry repo, and **schedules KMS key-version destruction** (Cloud KMS enforces a mandatory 24-hour reversible delay — this is a compliance feature, not a bug).
8. **Post-cleanup verification** — re-scans and prints anything still alive.

### Manual Console Double-Check (Recommended Before You Log Off)

```bash
# Confirm zero endpoints are left running (this is the one that costs real money):
gcloud ai endpoints list --region="${REGION}" --project="${PROJECT_ID}"

# Confirm no training jobs are still burning GPU time:
gcloud ai custom-jobs list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING"

# Confirm no notebook runtime is idling:
gcloud colab runtimes list --region="${REGION}" --project="${PROJECT_ID}" 2>/dev/null
```

Expected output for all three: `Listed 0 items.`

> [!TIP]
> **Nuclear option for throwaway sandbox projects**: If you provisioned a dedicated Argolis sandbox project purely for this bootcamp, the cleanest teardown is to delete the entire project (`gcloud projects delete ${PROJECT_ID}`). This guarantees zero orphaned billing. **Only do this if the project contains nothing else you care about** — project deletion is irreversible after a 30-day recovery window.

---

## 9. Production Engineering Safeguards: Completion-Only Loss, OOD & Adversarial Red-Teaming, and 3-Way Baseline Gating

To guarantee enterprise readiness for Indian Banking, Financial Services, and Insurance (BFSI) deployments, this repository implements six additional production engineering safeguards:

1. **Completion-Only Loss Masking (`labels = -100` on Prompt & Padding)**:
   - Standard causal language model SFT computes cross-entropy loss across the entire sequence (including `<start_of_turn>user` instructions and system prompts), wasting gradient capacity on memorizing prompt phrasing.
   - [`training/train_gemma_qlora.py`](training/train_gemma_qlora.py) uses [`make_completion_only_collator`](training/train_gemma_qlora.py) to scan for the last `<start_of_turn>model\n` boundary token sequence and set `labels[i] = -100` on every preceding prompt token and trailing pad token (`PyTorch CrossEntropyLoss ignore_index = -100`), plus logs a decoded supervised-span preview and sequence truncation audit before step 1.
2. **Multimodal Gemma 4 Language-Backbone LoRA Regex (`LM_ALL_PROJ`)**:
   - Multimodal Gemma 4 (`google/gemma-4-12B-it`, `google/gemma-4-E4B-it`) includes `vision_tower` and `audio_tower` modules with linear projection layers (`q_proj`, `k_proj`, etc.). Using a naive list of target module names attaches LoRA adapters to frozen vision/audio towers during text-only SFT.
   - [`training/train_gemma_qlora.py`](training/train_gemma_qlora.py) uses `LM_ALL_PROJ = r"^model\.language_model\..*\.(q|k|v|o|gate|up|down)_proj$"` on multimodal models so only the language backbone receives LoRA adapters.
3. **5-Split Dataset Architecture + Cryptographic Data Card (`data/data_card.json`)**:
   - [`data/generate_fsi_dataset.py`](data/generate_fsi_dataset.py) generates 5 disjoint splits verified by [`data/inspect_dataset.py`](data/inspect_dataset.py):
     - `indian_fsi_sft_train.jsonl` (1,200 examples)
     - `indian_fsi_sft_val.jsonl` (150 examples)
     - `indian_fsi_sft_test.jsonl` (150 In-Distribution holdout examples)
     - `indian_fsi_sft_test_ood.jsonl` (150 Out-of-Distribution examples with unseen merchants, neo-banks, NBFCs, and Hinglish/code-mixed prompts)
     - `indian_fsi_sft_challenge.jsonl` (25 Adversarial prompt-injection & boundary stress-tests for PMLA Section 12A non-tipping-off and FLDG >5% caps)
   - [`data/data_card.json`](data/data_card.json) records SHA-256 digests, split counts, regulatory domains, and RBI/SEBI/IRDAI provenance.
4. **3-Way Benchmark Gating (`Rules Baseline` vs. `Untuned Base Gemma` vs. `Fine-Tuned FSI Gemma`)**:
   - Before promoting any fine-tuned checkpoint, [`evaluation/evaluate_fsi_model.py`](evaluation/evaluate_fsi_model.py) evaluates a deterministic Indian FSI Regex/Rules Engine alongside Untuned Base Gemma and Fine-Tuned FSI Gemma across In-Distribution (`test`), Out-of-Distribution (`test_ood`), and Adversarial (`challenge`) splits with 1,000-iteration bootstrap 95% confidence intervals.
5. **Dedicated Bank Statement / Passbook Narration Benchmark (`evaluation/passbook_parser_benchmark.py`)**:
   - [`evaluation/passbook_parser_benchmark.py`](evaluation/passbook_parser_benchmark.py) benchmarks structured 7-key JSON extraction (`date`, `rail`, `counterparty`, `txn_type`, `amount_inr`, `category`, `reference_id`) on noisy Indian bank passbook narrations (`DATE | NARRATION | DR/CR | AMOUNT`) across `test_id` and `test_ood`, reporting per-field accuracy and 12-class Category Macro-F1.
6. **Live OpenAI-Compatible Endpoint Evaluator & Cloud Run GPU Fast-Path (`evaluation/predict_endpoint.py`)**:
   - [`evaluation/predict_endpoint.py`](evaluation/predict_endpoint.py) scores live Vertex AI or Cloud Run GPU vLLM `/v1/chat/completions` endpoints concurrently with optional constrained decoding (`--json-schema`), `enable_thinking: False` for deterministic JSON output, and `p50`/`p95` latency reporting.
   - [`scripts/03_deploy_sovereign_vllm.sh`](scripts/03_deploy_sovereign_vllm.sh) supports Cloud Run GPU in-region GCS FUSE mounting (`--add-volume=...`), `--enable-prefix-caching`, and `--limit-mm-per-prompt={"image":0,"audio":0}` to skip multimodal encoder warmup.

---

## Repository Directory Structure

```text
gemma-finetuning/
├── README.md                                                    # Principal Architect Masterclass & Bootcamp Guide (this file)
├── requirements.txt                                             # Lightweight local dependencies
├── docs/
│   ├── 01_FUNDAMENTALS_FOR_BEGINNERS.md                         # 📘 Module 0: Plain-English LLM/fine-tuning theory, every equation decoded, 45-term glossary
│   ├── 02_END_TO_END_RUNBOOK.md                                 # 🧭 All 8 lab steps: commands, expected output, timings, costs, troubleshooting, cleanup
│   └── images/                                                  # Rendered architecture & concept diagrams (JPG)
│       ├── 01_sovereign_architecture.jpg                        # Sovereign VPC-SC / CMEK / Vertex AI reference architecture
│       ├── 02_prompting_vs_rag_vs_finetuning.jpg                # Decision guide: prompting vs RAG vs fine-tuning
│       ├── 03_lora_math_explained.jpg                           # LoRA low-rank decomposition visualised
│       ├── 04_qlora_memory_budget.jpg                           # 196 GB full FT vs 13.4 GB QLoRA memory budget
│       ├── 05_evals_hillclimbing_flywheel.jpg                   # The 4-stage evals & hillclimbing loop
│       └── 06_end_to_end_lab_journey.jpg                        # The 8-step end-to-end lab journey
├── data/
│   ├── generate_fsi_dataset.py                                  # 100% Stdlib Synthetic Indian FSI Dataset Generator (5 splits + data_card.json)
│   ├── inspect_dataset.py                                       # ML Best Practices EDA, Completion-Boundary & Zero-Leakage Auditor
│   ├── data_card.json                                           # Cryptographic SHA-256 manifest, split counts & regulatory provenance
│   ├── indian_fsi_sft_train.jsonl                               # 1,200 Stratified Training Instruction Pairs
│   ├── indian_fsi_sft_val.jsonl                                 # 150 Stratified Validation Instruction Pairs
│   ├── indian_fsi_sft_test.jsonl                                # 150 Holdout In-Distribution Test Instruction Pairs
│   ├── indian_fsi_sft_test_ood.jsonl                            # 150 Holdout Out-of-Distribution (OOD & Hinglish) Test Pairs
│   └── indian_fsi_sft_challenge.jsonl                           # 25 Adversarial Prompt-Injection & Regulatory Boundary Stress Cases
├── training/
│   ├── train_gemma_qlora.py                                     # Production 4-Bit NF4 QLoRA (Completion-Only Loss, LM_ALL_PROJ, /gcs/ FUSE, MODEL_CARD)
│   ├── requirements-train.txt                                   # GPU container dependencies (TRL, PEFT, BitsAndBytes)
│   └── Dockerfile                                               # Remote Cloud Build container definition (never built locally)
├── scripts/
│   ├── 00_preflight_check.sh                                    # ✅ Read-only 7-point validator: auth, ADC, billing, APIs, GPU quota, HF token
│   ├── 01_setup_sovereign_infra.sh                              # Provisions KMS CMEK, GCS & Artifact Registry in asia-south1
│   ├── 02_submit_vertex_finetuning.sh                           # Zero-Docker Vertex AI Custom Job Launcher (Prebuilt DLC + Cloud Build + Base Mirror)
│   ├── 03_deploy_sovereign_vllm.sh                              # Deploys merged BF16 model to vLLM on Vertex AI / Cloud Run GPU / Model Garden
│   ├── 04_test_endpoint.sh                                      # 🔬 Smoke test: 3 real FSI prompts against the live endpoint + latency
│   └── 99_cleanup_resources.sh                                  # 🧹 Teardown (DRY-RUN by default; --confirm / --keep-bucket)
├── evaluation/
│   ├── evaluate_fsi_model.py                                    # 3-Way Holdout Suite (Rules vs Base vs Tuned across ID, OOD, Challenge + 95% CIs)
│   ├── passbook_parser_benchmark.py                             # Indian Bank Passbook Narration 7-Key JSON Schema & Category Macro-F1 Benchmark
│   ├── predict_endpoint.py                                      # Concurrent OpenAI-Compatible Live Endpoint Scorer (--json-schema, p50/p95 latency)
│   └── hillclimbing_simulator.py                                # 6-Iteration Evals & Hillclimbing Flywheel Simulator
└── notebooks/
    └── Gemma_FSI_Sovereign_FineTuning_Bootcamp.ipynb            # Interactive 15-Cell Vertex AI Workbench / Colab Enterprise Bootcamp Notebook
```
