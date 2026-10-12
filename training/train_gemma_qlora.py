#!/usr/bin/env python3
"""
Gemma 4 / Gemma 3 / Gemma 2 QLoRA Fine-Tuning Script for Indian FSI Sovereignty
===============================================================================
Designed for execution on Google Cloud Vertex AI Custom Training or Vertex AI Workbench
in `asia-south1` (Mumbai) / `asia-south2` (Delhi NCR).

Supported Open-Weight Model Generations:
1. GEMMA 4 (Latest 2026 Flagship Generation):
   - `google/gemma-4-26b-a4b-it`: Mixture-of-Experts (MoE) — 26B total params, only 4B active per token!
     Delivers 26B-level RBI/SEBI regulatory intelligence at 4B inference speed/cost on 1x NVIDIA L4.
   - `google/gemma-4-31b-it`: Dense 31B flagship model for deep financial/legal reasoning.
   - `google/gemma-4-12B-it` & `google/gemma-4-E4B-it`: Gemma 4 Unified instruction-tuned models.
2. GEMMA 3 (Long-Context 128K & Multimodal Generation):
   - `google/gemma-3-12b-it` (Default single-L4 sweet spot) & `google/gemma-3-27b-it` (128K context window).
3. GEMMA 2 (Foundational Dense Generation):
   - `google/gemma-2-9b-it` & `google/gemma-2-27b-it`.

Key Technical Highlights:
- Completion-Only Loss Masking (`-100` on system/user prompt and right-padding tokens) so loss is
  computed strictly on the assistant's regulatory response, with decoded supervised-span preview.
- Automatic MoE, Multimodal Language-Tower (`LM_ALL_PROJ`), & Dense target module detection for LoRA.
- 4-bit NormalFloat (NF4) Quantization via `bitsandbytes` with double quantization & `bfloat16` compute.
- Native Vertex AI `/gcs/` Cloud Storage FUSE fast-path + optional In-Region Base Model Mirroring (`--mirror-to`).
- Structured single-line JSON logging (`JsonLogs`) for Cloud Logging & automatic `MODEL_CARD.md` + `run_config.json` paper trail.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Union

# Regex targeting all 7 linear projections of the language model tower only (leaves vision/audio towers untouched)
LM_ALL_PROJ = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def fuse_gcs_path(uri: Optional[str]) -> Optional[str]:
    """Translates gs://bucket/path -> /gcs/bucket/path when Vertex AI Cloud Storage FUSE is mounted."""
    if uri and uri.startswith("gs://") and os.path.isdir("/gcs"):
        return "/gcs/" + uri[len("gs://"):]
    return uri


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_json(msg: str, **kv) -> None:
    """Emits structured single-line JSON logs for Google Cloud Logging."""
    print(json.dumps({"t": time.strftime("%H:%M:%S"), "msg": msg, **kv}), flush=True)


def copy_aux_files(src: str, dst: Path) -> None:
    """Carry over processor/chat-template configs so vLLM and Model Garden can load the merged checkpoint."""
    src_path = Path(src)
    if not src_path.is_dir():
        return
    for f in src_path.iterdir():
        if f.is_file() and not f.name.endswith((".safetensors", ".bin", ".index.json")) and not (dst / f.name).exists():
            shutil.copy2(f, dst / f.name)


def download_from_gcs_if_needed(gcs_or_local_path: str, local_target: str) -> str:
    """Resolves a GCS URI via Vertex AI /gcs/ FUSE mount if available, otherwise via gcloud storage cp."""
    if not gcs_or_local_path.startswith("gs://"):
        return gcs_or_local_path
    fused = fuse_gcs_path(gcs_or_local_path)
    if fused and os.path.exists(fused):
        print(f"[*] Using Vertex AI Cloud Storage FUSE mount: {fused}")
        return fused
    os.makedirs(os.path.dirname(local_target), exist_ok=True)
    print(f"[*] Downloading {gcs_or_local_path} -> {local_target}...")
    subprocess.run(["gcloud", "storage", "cp", gcs_or_local_path, local_target], check=True)
    return local_target


def upload_to_gcs_if_needed(local_dir: str, gcs_target_uri: str) -> None:
    """Uploads local directory contents to a GCS URI (via /gcs/ FUSE or gcloud storage cp)."""
    if not gcs_target_uri.startswith("gs://"):
        return
    fused = fuse_gcs_path(gcs_target_uri)
    if fused and os.path.isdir("/gcs"):
        print(f"[*] Copying artifacts via Cloud Storage FUSE {local_dir} -> {fused}...")
        Path(fused).mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_dir, fused, dirs_exist_ok=True)
        return
    print(f"[*] Uploading fine-tuned model artifacts {local_dir} -> {gcs_target_uri}...")
    subprocess.run(["gcloud", "storage", "cp", "-r", f"{local_dir}/*", gcs_target_uri], check=True)


def get_lora_target_modules(model_id: str, is_multimodal: bool = False) -> Union[List[str], str]:
    """
    Returns optimal LoRA target linear modules based on Gemma architecture generation
    (Dense Gemma 2/3/4 vs. Mixture-of-Experts Gemma 4 26B-A4B vs. Multimodal Language Tower).
    """
    base_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    if "a4b" in model_id.lower() or "moe" in model_id.lower():
        print("[*] Detected Gemma 4 Mixture-of-Experts (MoE) architecture (`26b-a4b`).")
        print("    Targeting attention projections + active 4B MoE expert linear layers.")
        return base_modules + ["router", "gate"]
    if is_multimodal:
        print("[*] Detected Multimodal Gemma architecture — scoping LoRA strictly to language_model linear layers.")
        return LM_ALL_PROJ
    return base_modules


def find_last_subsequence(seq: List[int], pattern: List[int]) -> Optional[int]:
    """Returns the index immediately after the last occurrence of `pattern` in `seq`."""
    if not pattern:
        return None
    n = len(pattern)
    for i in range(len(seq) - n, -1, -1):
        if seq[i : i + n] == pattern:
            return i + n
    return None


def make_completion_only_collator(tokenizer, max_length: int):
    """
    Tokenizes chats and trains ONLY on the assistant's regulatory response.
    Everything up to and including `<start_of_turn>model\\n` (Gemma 2/3) or `<|turn>model\\n` (Gemma 4),
    plus right-padding positions (`attention_mask == 0`), receives label `-100`.
    """
    markers = []
    for header in ("<start_of_turn>model\n", "<|turn>model\n"):
        encoded_hdr = tokenizer.encode(header, add_special_tokens=False)
        if encoded_hdr:
            markers.append(encoded_hdr)

    stats = {"batches": 0, "dropped": 0, "truncated": 0}

    def collate(examples):
        texts = []
        for ex in examples:
            if "gemma_text" in ex and ex["gemma_text"]:
                texts.append(ex["gemma_text"].strip())
            else:
                texts.append(
                    tokenizer.apply_chat_template(
                        ex["messages"], tokenize=False, enable_thinking=False
                    ).strip()
                )
        tokenizer.padding_side = "right"
        batch = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        for i in range(labels.size(0)):
            row = batch["input_ids"][i].tolist()
            if int(batch["attention_mask"][i].sum()) >= max_length:
                stats["truncated"] += 1
            start_idx = None
            for m in markers:
                start_idx = find_last_subsequence(row, m)
                if start_idx is not None:
                    break
            if start_idx is None:
                labels[i, :] = -100  # Never train on unmasked prompt tokens if delimiter is missing
                stats["dropped"] += 1
            else:
                labels[i, :start_idx] = -100
        # Position-based padding mask (do not mask by pad_token_id value so valid EOS tokens remain supervised)
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        stats["batches"] += 1
        return batch

    def preview_supervised_span(example) -> str:
        b = collate([example])
        active_ids = [
            t for t, l in zip(b["input_ids"][0].tolist(), b["labels"][0].tolist()) if l != -100
        ]
        return tokenizer.decode(active_ids, skip_special_tokens=False)

    collate.stats = stats
    collate.preview = preview_supervised_span
    return collate


def generate_model_card(args, data_hashes: dict, collate_stats: dict, train_secs: float, gpu_name: str) -> str:
    return f"""# Sovereign Indian FSI Model Card: `{args.model_id}` (QLoRA)

| Attribute | Value |
| :--- | :--- |
| **Base Model** | `{args.model_id}` |
| **Fine-Tuning Method** | 4-Bit NF4 QLoRA (`bfloat16` compute, double quantization); `r={args.lora_r}`, `alpha={args.lora_alpha}`, `dropout={args.lora_dropout}` |
| **Completion-Only Loss Masking** | Enabled (`-100` on system/user prompt & padding; `truncated={collate_stats['truncated']}`, `dropped={collate_stats['dropped']}`) |
| **Training Hyperparameters** | `epochs={args.epochs}`, `lr={args.learning_rate}`, `batch={args.batch_size}x{args.grad_accum}` (effective `{args.batch_size * args.grad_accum}`), `max_seq_length={args.max_seq_length}` |
| **Hardware & Wall-Clock Time** | `{train_secs / 60:.1f} min` on `{gpu_name}` (`asia-south1` / `asia-south2`) |
| **Created (UTC)** | `{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}` |
| **Runtime Environment** | Python `{platform.python_version()}` |

## Intended Regulatory & Operational Use
Domain-adapted assistant for Indian Banking, Financial Services, and Insurance (BFSI) compliance workflows:
1. **RBI KYC / AML & PMLA**: 10% UBO rule, 2/8/10-year Re-KYC cycles, 7-day FIU-IND STR timeline, and strict Section 12A non-tipping-off enforcement.
2. **NPCI UPI UDIR Disputes**: RBI TAT Circular (`RBI/2019-20/67`) T+1 / T+5 turnaround times and proactive `₹100/day` compensation math.
3. **RBI Digital Lending (DLG & KFS)**: All-inclusive APR calculation with 18% GST, 5% FLDG cap enforcement, and direct RE-to-borrower fund flow checks.
4. **SEBI Capital Markets**: Peak Margin intraday snapshot shortfall penalties and Algo Trading API OTR governance.
5. **ISO-20022 Payments (`pacs.008` / `pacs.002`)**: RTGS/NEFT rejection diagnostics and `₹50 Crore` 20-character LEI verification.

## Dataset Provenance (SHA-256 Lineage)
- `train`: `{data_hashes.get('train', 'n/a')}`
- `validation`: `{data_hashes.get('validation', 'n/a')}`

## Governance & Limitations
- Decision-support and compliance drafting assistant; requires human reviewer sign-off before executing ledger entries or regulatory filings.
- All outputs should be validated with deterministic Tier-1 citation/math graders (`evaluation/evaluate_fsi_model.py`).
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-Tune Gemma 4 / Gemma 3 / Gemma 2 with QLoRA for Indian FSI")
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-4-12B-it",
        help=(
            "HuggingFace Model ID or GCS URI. Supported models (Gemma 4 default):\n"
            "  - google/gemma-4-12B-it     (Gemma 4 12B Unified - Default)\n"
            "  - google/gemma-4-E4B-it     (Gemma 4 E4B Effective 4B - Fast single-L4)\n"
            "  - google/gemma-4-26B-A4B-it (Gemma 4 MoE: 26B total / 4B active)\n"
            "  - google/gemma-4-31B-it     (Gemma 4 Dense 31B Flagship)\n"
            "  - google/gemma-3-12b-it     (Gemma 3 12B 128K Context)\n"
            "  - google/gemma-3-27b-it     (Gemma 3 27B 128K Context)\n"
            "  - google/gemma-2-9b-it      (Gemma 2 9B Dense)"
        ),
    )
    parser.add_argument(
        "--mirror-to",
        type=str,
        default="",
        help="Optional gs:// folder to mirror Hugging Face Hub base weights into Mumbai GCS for zero-Hub future runs",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="data/indian_fsi_sft_train.jsonl",
        help="Local path or GCS URI (gs://...) to training JSONL dataset",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default="data/indian_fsi_sft_val.jsonl",
        help="Local path or GCS URI (gs://...) to validation JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("AIP_MODEL_DIR", "./output/gemma-fsi-qlora"),
        help="Local directory or GCS URI (gs://...) for saving fine-tuned weights",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (r). Recommended: 16 or 32 for FSI domain adaptation")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor (typically 2 * lora_r)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum token sequence length (up to 128K on Gemma 3/4)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device training batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (effective batch size = batch * accum)")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Peak learning rate for AdamW paged optimizer")
    parser.add_argument("--merge-weights", action="store_true", help="Merge LoRA adapter into base model and save full BF16 safetensors")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    class JsonLogs(TrainerCallback):
        """Prints one structured JSON line per log event so Cloud Logging shows a clean loss curve."""

        def on_log(self, cb_args, state, control, logs=None, **kwargs):
            if logs:
                log_json(
                    "train",
                    step=state.global_step,
                    epoch=round(state.epoch or 0.0, 3),
                    **{k: round(v, 5) if isinstance(v, float) else v for k, v in logs.items()},
                )

    local_train = download_from_gcs_if_needed(args.train_dataset, "/tmp/data/train.jsonl")
    local_val = download_from_gcs_if_needed(args.val_dataset, "/tmp/data/val.jsonl")
    data_hashes = {
        "train": sha256_file(local_train) if os.path.exists(local_train) else "n/a",
        "validation": sha256_file(local_val) if os.path.exists(local_val) else "n/a",
    }

    is_gcs_output = args.output_dir.startswith("gs://")
    local_output_dir = "/tmp/output/gemma-fsi-qlora" if is_gcs_output else args.output_dir
    os.makedirs(local_output_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")

    # Resolve base model from GCS mirror or Hugging Face Hub
    model_source = fuse_gcs_path(args.model_id) or args.model_id
    if model_source.startswith("/gcs/") and os.path.exists(model_source):
        local_base_dir = "/tmp/base_model"
        log_json("copying base weights from Cloud Storage FUSE mirror", source=model_source)
        shutil.copytree(model_source, local_base_dir, dirs_exist_ok=True)
        model_source = local_base_dir
    elif args.mirror_to:
        from huggingface_hub import snapshot_download
        local_base_dir = "/tmp/base_model"
        snapshot_download(model_source, local_dir=local_base_dir, token=hf_token)
        upload_to_gcs_if_needed(local_base_dir, args.mirror_to)
        log_json("mirrored base model weights to GCS", mirror_to=args.mirror_to)
        model_source = local_base_dir

    print(f"[*] Loading Tokenizer for Base Model: {model_source}")
    tokenizer = AutoTokenizer.from_pretrained(model_source, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[*] Configuring 4-bit NF4 BitsAndBytes Quantization (Compute Dtype: bfloat16)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[*] Loading Base Model `{model_source}` onto GPU...")
    is_multimodal = False
    try:
        from transformers import AutoModelForMultimodalLM
        if any(tag in args.model_id.lower() for tag in ("gemma-4-12b", "gemma-4-e4b")):
            model = AutoModelForMultimodalLM.from_pretrained(
                model_source,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                token=hf_token,
            )
            is_multimodal = True
        else:
            raise ValueError("Use CausalLM loader")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            token=hf_token,
        )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    target_modules = get_lora_target_modules(args.model_id, is_multimodal=is_multimodal)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    print("[*] Loading Indian FSI Train & Validation JSONL Datasets...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": local_train,
            "validation": local_val,
        },
    )

    collator = make_completion_only_collator(tokenizer, args.max_seq_length)
    log_json(
        "verified completion-only loss mask on train[0]",
        supervised_preview=collator.preview(dataset["train"][0])[:240],
    )

    sft_config = SFTConfig(
        output_dir=local_output_dir,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[JsonLogs()],
        args=sft_config,
    )

    trainer.model.print_trainable_parameters()
    print(f"[*] Starting QLoRA Supervised Fine-Tuning (SFT) on {args.model_id}...")
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    adapter_save_path = os.path.join(local_output_dir, "lora_adapter")
    print(f"[+] Saving LoRA Adapter weights to {adapter_save_path}...")
    trainer.model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    log_json("training complete", minutes=round(train_secs / 60.0, 2), collator_stats=collator.stats)

    if args.merge_weights:
        merged_save_path = os.path.join(local_output_dir, "merged_model_bf16")
        print(f"[*] Merging LoRA Adapter into Base Model (BF16) -> {merged_save_path}...")
        del model
        del trainer
        torch.cuda.empty_cache()

        base_model_bf16 = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )
        merged_model = PeftModel.from_pretrained(base_model_bf16, adapter_save_path)
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(merged_save_path, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(merged_save_path)
        copy_aux_files(model_source, Path(merged_save_path))
        print(f"[+] Standalone BF16 Merged Model saved to {merged_save_path}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    card_md = generate_model_card(args, data_hashes, collator.stats, train_secs, gpu_name)
    Path(local_output_dir, "MODEL_CARD.md").write_text(card_md, encoding="utf-8")
    Path(local_output_dir, "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "data_sha256": data_hashes,
                "collator_stats": collator.stats,
                "train_seconds": round(train_secs, 2),
                "gpu": gpu_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if is_gcs_output:
        upload_to_gcs_if_needed(local_output_dir, args.output_dir)

    print("[+] Fine-Tuning Pipeline & Model Card Generation Completed Successfully!")


if __name__ == "__main__":
    main()
