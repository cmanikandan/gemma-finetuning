#!/usr/bin/env python3
"""
Live Sovereign vLLM Endpoint Scorer & Constrained Decoding Benchmark
====================================================================
Evaluates a deployed Cloud Run GPU or Vertex AI vLLM endpoint over its
OpenAI-compatible `/v1/chat/completions` API using only the Python Standard Library.

Supports:
- Cloud Run + vLLM via local proxy:
    `gcloud run services proxy gemma-indian-fsi-sovereign --region=asia-south1 --port=9090`
    `python3 evaluation/predict_endpoint.py --base-url http://localhost:9090/v1 --test-file data/indian_fsi_sft_test_ood.jsonl`
- Vertex AI Dedicated Endpoint with Bearer Token:
    `python3 evaluation/predict_endpoint.py --base-url https://<ENDPOINT_DNS>/v1 --token "$(gcloud auth print-access-token)"`
- Optional JSON Schema Constrained Decoding (`--json-schema`) for structured extraction tasks.
- Concurrent worker pool (`--workers 8`) reporting p50 / p95 latency and throughput (req/s).
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from evaluate_fsi_model import bootstrap_ci, check_pmla_and_override_safety, evaluate_prediction, load_jsonl

FSI_DETERMINATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {
            "type": "string",
            "enum": [
                "RBI_KYC_AML",
                "NPCI_UPI_DISPUTE",
                "DIGITAL_LENDING_KFS",
                "SEBI_CAPITAL_MARKETS",
                "ISO20022_PAYMENTS_OPS",
            ],
        },
        "primary_circular_citation": {"type": "string"},
        "computed_financial_figure_inr_or_pct": {"type": "string"},
        "compliance_status": {"type": "string", "enum": ["COMPLIANT", "NON_COMPLIANT", "REQUIRES_STR_OR_REVERSAL"]},
        "operational_runbook_summary": {"type": "string"},
    },
    "required": [
        "domain",
        "primary_circular_citation",
        "computed_financial_figure_inr_or_pct",
        "compliance_status",
        "operational_runbook_summary",
    ],
    "additionalProperties": False,
}


def call_openai_compatible_vllm(
    base_url: str,
    model: str,
    token: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    use_json_schema: bool = False,
) -> Tuple[str, float]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict = {
        "model": model,
        "messages": messages[:2],  # Strictly system + user; never leak ground-truth assistant answer
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if use_json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "fsi_regulatory_determination", "schema": FSI_DETERMINATION_JSON_SCHEMA},
        }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    content = body["choices"][0]["message"].get("content") or ""
    return content.strip(), latency_ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-file", default="data/indian_fsi_sft_test_ood.jsonl", help="JSONL split to evaluate")
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible vLLM base URL (e.g. http://localhost:9090/v1)")
    ap.add_argument("--model", default="gemma-indian-fsi-sovereign", help="Served model name in vLLM")
    ap.add_argument("--token", default="EMPTY", help="Bearer token (or EMPTY for local Cloud Run proxy)")
    ap.add_argument("--limit", type=int, default=0, help="Max rows to score (0 = full split)")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent request workers")
    ap.add_argument("--json-schema", action="store_true", help="Enable vLLM JSON-Schema constrained decoding")
    ap.add_argument("--out", default="runs/preds_endpoint.jsonl", help="Output JSONL path for predictions")
    args = ap.parse_args()

    rows = load_jsonl(args.test_file)
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"[*] Scoring {len(rows)} examples against {args.base_url} (model={args.model}, workers={args.workers})...")
    t_start = time.time()

    def _worker(row: Dict) -> Tuple[str, float]:
        return call_openai_compatible_vllm(
            base_url=args.base_url,
            model=args.model,
            token=args.token,
            messages=row["messages"],
            use_json_schema=args.json_schema,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_worker, rows))

    wall_secs = time.time() - t_start
    preds = [p for p, _ in results]
    latencies = sorted(ms for _, ms in results)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"id": r["id"], "domain": r["domain"], "prediction": p}, ensure_ascii=False) + "\n")

    comp_scores = []
    safety_scores = []
    for r, p in zip(rows, preds):
        m = evaluate_prediction(p, r["assistant_response"])
        comp_scores.append(m["composite_score"])
        safety_scores.append(check_pmla_and_override_safety(r["user_prompt"], p))

    mean_c, lo_c, hi_c = bootstrap_ci(comp_scores)
    p50_ms = statistics.median(latencies) if latencies else 0.0
    p95_ms = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0

    print("=" * 88)
    print(f" LIVE ENDPOINT EVALUATION REPORT: {args.test_file} -> {args.base_url}")
    print("=" * 88)
    print(f"  * Examples Evaluated        : {len(rows)}")
    print(f"  * Composite Score (95% CI)  : {mean_c*100:5.1f}% [{lo_c*100:4.1f}%, {hi_c*100:4.1f}%]")
    print(f"  * PMLA Non-Tipping-Off Safe : {sum(safety_scores)/len(safety_scores)*100:5.1f}%")
    print(f"  * Latency (p50 / p95)       : p50 = {p50_ms:.0f} ms | p95 = {p95_ms:.0f} ms")
    print(f"  * Throughput                : {len(rows) / max(wall_secs, 0.001):.1f} req/s ({args.workers} workers)")
    print(f"  * Predictions Saved To      : {args.out}")
    print("=" * 88)


if __name__ == "__main__":
    main()
