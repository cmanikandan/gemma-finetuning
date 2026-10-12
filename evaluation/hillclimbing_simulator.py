#!/usr/bin/env python3
"""
Principal Architect Hillclimbing & Evals Flywheel Simulator (Indian FSI)
========================================================================
Demonstrates how AI Engineering teams use Evaluation-Driven Development (EDD)
and iterative "Hillclimbing" to systematically optimize an open-weight Gemma model
from a 14.2% zero-shot baseline to 99.9% RBI/SEBI regulatory compliance.

Runs with 100% pure Python Standard Library — Zero local Docker or pip packages required.
"""

from typing import Dict, List


HILLCLIMBING_ITERATIONS: List[Dict] = [
    {
        "version": "v0: Base Model (Zero-Shot)",
        "model_id": "google/gemma-4-12B-it (Untuned)",
        "intervention": "Out-of-the-box instruction model with standard system prompt.",
        "slice_scores": {
            "RBI_KYC_AML": 16.5,
            "NPCI_UPI_DISPUTE": 12.0,
            "DIGITAL_LENDING_KFS": 11.4,
            "SEBI_CAPITAL_MARKETS": 15.8,
            "ISO20022_PAYMENTS_OPS": 15.3,
        },
        "citation_recall": 0.0,
        "apr_math_acc": 12.0,
        "pmla_non_tipping_off": 68.0,
        "llm_judge_rubric_5pt": 2.1,
        "ttft_ms": 22.0,
        "vram_gb": 13.4,
        "root_cause_diagnosis": (
            "Model speaks fluent English but cites US/EU laws (FDIC/Reg E) instead of RBI/NPCI circulars; "
            "fails 18% GST deduction in KFS APR math."
        ),
    },
    {
        "version": "v1: Prompt Eng + Few-Shot RAG",
        "model_id": "google/gemma-4-12B-it + Vertex AI Search RAG",
        "intervention": "Injected top-3 RBI/SEBI circular chunks + 3-shot examples into 8K prompt window.",
        "slice_scores": {
            "RBI_KYC_AML": 64.2,
            "NPCI_UPI_DISPUTE": 55.8,
            "DIGITAL_LENDING_KFS": 48.0,
            "SEBI_CAPITAL_MARKETS": 61.5,
            "ISO20022_PAYMENTS_OPS": 62.5,
        },
        "citation_recall": 62.0,
        "apr_math_acc": 44.0,
        "pmla_non_tipping_off": 74.0,
        "llm_judge_rubric_5pt": 3.2,
        "ttft_ms": 85.0,
        "vram_gb": 16.8,
        "root_cause_diagnosis": (
            "RAG improves citation recall (+62%), but prompt bloat increases TTFT latency by 3.8x (85ms) "
            "and model still fails multi-step KFS APR math and structured CBS XML runbooks."
        ),
    },
    {
        "version": "v2: First QLoRA SFT (r=8, Attention Only)",
        "model_id": "google/gemma-4-12B-it (QLoRA r=8, q_proj/v_proj)",
        "intervention": "1 Epoch SFT on 1,200 Indian FSI examples; LoRA applied ONLY to q_proj & v_proj.",
        "slice_scores": {
            "RBI_KYC_AML": 79.4,
            "NPCI_UPI_DISPUTE": 74.2,
            "DIGITAL_LENDING_KFS": 71.0,
            "SEBI_CAPITAL_MARKETS": 78.5,
            "ISO20022_PAYMENTS_OPS": 79.4,
        },
        "citation_recall": 81.0,
        "apr_math_acc": 68.5,
        "pmla_non_tipping_off": 82.0,
        "llm_judge_rubric_5pt": 3.9,
        "ttft_ms": 23.0,
        "vram_gb": 12.1,
        "root_cause_diagnosis": (
            "Eliminated RAG prompt latency (back to 23ms TTFT!). However, freezing MLP layers (gate/up/down_proj) "
            "caused underfitting on numerical APR calculations and ISO-20022 pacs.008 error codes."
        ),
    },
    {
        "version": "v3: Architecture Climb (r=16, All 7 Linear Modules)",
        "model_id": "google/gemma-4-12B-it (QLoRA r=16, All Linear + bfloat16)",
        "intervention": "Targeted all 7 linear projections (q/k/v/o + gate/up/down_proj), 3 Epochs, Cosine LR.",
        "slice_scores": {
            "RBI_KYC_AML": 84.5,  # Notice slice lag!
            "NPCI_UPI_DISPUTE": 94.8,
            "DIGITAL_LENDING_KFS": 93.2,
            "SEBI_CAPITAL_MARKETS": 92.6,
            "ISO20022_PAYMENTS_OPS": 94.1,
        },
        "citation_recall": 94.0,
        "apr_math_acc": 93.5,
        "pmla_non_tipping_off": 79.0,  # SLICE REGRESSION DETECTED!
        "llm_judge_rubric_5pt": 4.4,
        "ttft_ms": 23.0,
        "vram_gb": 13.4,
        "root_cause_diagnosis": (
            "CRITICAL EVAL DISCOVERY (Slice Regression): Overall score jumped to 91.8%, but PMLA Non-Tipping-Off "
            "regressed from 82% -> 79%! Model became 'too helpful' and explained AML freeze reasons to customers "
            "(violating Section 12A PMLA) and occasionally cited old 25% UBO rule instead of 2023 10% rule."
        ),
    },
    {
        "version": "v4: Data-Centric Climb (Targeted Hard Negatives)",
        "model_id": "google/gemma-4-12B-it (QLoRA r=16 + 150 Hard Negatives)",
        "intervention": "Added 150 targeted synthetic examples focusing on PMLA Sec 12A non-tipping-off & 10% UBO edge cases.",
        "slice_scores": {
            "RBI_KYC_AML": 98.2,
            "NPCI_UPI_DISPUTE": 98.6,
            "DIGITAL_LENDING_KFS": 98.4,
            "SEBI_CAPITAL_MARKETS": 98.1,
            "ISO20022_PAYMENTS_OPS": 98.9,
        },
        "citation_recall": 99.1,
        "apr_math_acc": 98.8,
        "pmla_non_tipping_off": 98.5,
        "llm_judge_rubric_5pt": 4.85,
        "ttft_ms": 23.0,
        "vram_gb": 13.4,
        "root_cause_diagnosis": (
            "Targeted data augmentation fixed the RBI_KYC_AML slice (+13.7% jump!) without degrading any other slice. "
            "Remaining 1.5% errors are subtle tone/refusal edge cases."
        ),
    },
    {
        "version": "v5: Alignment Climb (Gemma 4 MoE + DPO Guardrails)",
        "model_id": "google/gemma-4-26b-a4b-it (QLoRA SFT + 1-Epoch DPO)",
        "intervention": "Upgraded base to Gemma 4 MoE (26B total / 4B active) + Direct Preference Optimization (DPO) on refusal pairs.",
        "slice_scores": {
            "RBI_KYC_AML": 99.9,
            "NPCI_UPI_DISPUTE": 99.9,
            "DIGITAL_LENDING_KFS": 100.0,
            "SEBI_CAPITAL_MARKETS": 99.9,
            "ISO20022_PAYMENTS_OPS": 100.0,
        },
        "citation_recall": 100.0,
        "apr_math_acc": 100.0,
        "pmla_non_tipping_off": 100.0,
        "llm_judge_rubric_5pt": 4.98,
        "ttft_ms": 17.5,  # Faster TTFT due to 4B active MoE parameters!
        "vram_gb": 18.2,
        "root_cause_diagnosis": (
            "PARETO OPTIMUM ACHIEVED: 99.9% composite score across all 5 RBI/SEBI/NPCI slices, 100% PMLA compliance, "
            "and TTFT latency dropped to 17.5ms because only 4B active MoE parameters execute per token!"
        ),
    },
]


def print_hillclimbing_report():
    print("=" * 115)
    print(" PRINCIPAL ARCHITECT EVALS & HILLCLIMBING LEDGER — INDIAN FSI SOVEREIGN GEMMA OPTIMIZATION")
    print(" Objective: Maximize RBI/SEBI/NPCI Regulatory Composite Score subject to TTFT < 30ms & Single L4 GPU (24GB VRAM)")
    print("=" * 115)

    # Table 1: Overall Progression & System Metrics
    header = (
        f"{'Hillclimb Iteration':<38} | {'Composite':>9} | {'Citation':>8} | "
        f"{'APR Math':>8} | {'PMLA Safe':>9} | {'Judge (1-5)':>11} | {'TTFT':>7} | {'VRAM':>6}"
    )
    print("\n[1] MACRO HILLCLIMBING PROGRESSION & PARETO FRONTIER METRICS")
    print("-" * 115)
    print(header)
    print("-" * 115)

    for it in HILLCLIMBING_ITERATIONS:
        comp = sum(it["slice_scores"].values()) / len(it["slice_scores"])
        print(
            f"{it['version']:<38} | {comp:>8.1f}% | {it['citation_recall']:>7.1f}% | "
            f"{it['apr_math_acc']:>7.1f}% | {it['pmla_non_tipping_off']:>8.1f}% | "
            f"{it['llm_judge_rubric_5pt']:>9.2f}/5 | {it['ttft_ms']:>5.1f}ms | {it['vram_gb']:>4.1f}GB"
        )
    print("-" * 115)

    # Table 2: Slice-by-Slice Heatmap (Detecting Slice Regressions)
    print("\n[2] SLICE-BY-SLICE REGULATORY DOMAIN MATRIX (DETECTING 'WHACK-A-MOLE' REGRESSIONS)")
    print("-" * 115)
    s_header = (
        f"{'Hillclimb Iteration':<38} | {'RBI_KYC_AML':>12} | {'NPCI_UPI':>10} | "
        f"{'DLG_KFS':>10} | {'SEBI_CM':>10} | {'ISO20022':>10}"
    )
    print(s_header)
    print("-" * 115)
    for it in HILLCLIMBING_ITERATIONS:
        sc = it["slice_scores"]
        kyc_flag = " (!)" if it["version"].startswith("v3") else "    "
        print(
            f"{it['version']:<38} | {sc['RBI_KYC_AML']:>8.1f}%{kyc_flag} | {sc['NPCI_UPI_DISPUTE']:>9.1f}% | "
            f"{sc['DIGITAL_LENDING_KFS']:>9.1f}% | {sc['SEBI_CAPITAL_MARKETS']:>9.1f}% | {sc['ISO20022_PAYMENTS_OPS']:>9.1f}%"
        )
    print("-" * 115)
    print("  Note: (!) in v3 highlights the classic 'Slice Regression' caught by disaggregated Evals:")
    print("        While overall accuracy climbed from 76.5% -> 91.8%, PMLA Non-Tipping-Off safety dropped from 82% -> 79%.")
    print("        Without disaggregated slice Evals, v3 would have been deployed with a fatal RBI compliance vulnerability.")

    # Section 3: Detailed Architectural Diagnosis per Step
    print("\n[3] ITERATION-BY-ITERATION ARCHITECTURAL DIAGNOSIS & INTERVENTION LOG")
    print("-" * 115)
    for it in HILLCLIMBING_ITERATIONS:
        print(f"  * {it['version']} [{it['model_id']}]")
        print(f"    - Intervention : {it['intervention']}")
        print(f"    - Eval Finding : {it['root_cause_diagnosis']}\n")
    print("=" * 115)


if __name__ == "__main__":
    print_hillclimbing_report()
