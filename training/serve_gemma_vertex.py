#!/usr/bin/env python3
"""
Sovereign Gemma 4 / 3 GPU Inference Server for Vertex AI Endpoints & Cloud Run
==============================================================================
Runs inside the pre-built `gemma-qlora-trainer:v1` container (PyTorch 2.13 + CUDA 12.6
+ transformers 5.19.0 + bitsandbytes 0.50.2) on 1x NVIDIA L4 (24GB VRAM).

Features:
1. Downloads merged model shards from `AIP_STORAGE_URI` (`gs://...`) in parallel using
   the GCE metadata server token (or `/gcs/` FUSE mount if present).
2. Loads Gemma 4 12B / E4B / 26B-A4B onto the L4 GPU in 4-bit NF4 (`bfloat16` compute)
   so 12B weights fit comfortably in ~8.5 GB VRAM with zero OOM risk.
3. Exposes:
   - `GET  /health`              (Vertex AI `AIP_HEALTH_ROUTE`)
   - `POST /predict`             (Vertex AI `AIP_PREDICT_ROUTE`: `{"instances": [...]}` -> `{"predictions": [...]}`)
   - `POST /v1/completions`      (OpenAI-compatible completions)
   - `POST /v1/chat/completions` (OpenAI-compatible chat completions)
"""

import gc
import json
import os
import shutil
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = None
TOKENIZER = None
MODEL_READY_EVENT = threading.Event()
LOAD_ERROR = None
GENERATE_LOCK = threading.Lock()


def log(msg: str, **kv) -> None:
    print(json.dumps({"t": time.strftime("%H:%M:%S"), "msg": msg, **kv}), flush=True)


def get_gcp_token() -> str:
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def pick_scratch_dir() -> str:
    candidates = ["/var/tmp", "/tmp", "/dev/shm"]
    best = "/tmp"
    best_free = -1
    for c in candidates:
        if os.path.exists(c):
            free = shutil.disk_usage(c).free
            if free > best_free:
                best_free = free
                best = c
    target = os.path.join(best, "served_model")
    log("selected scratch directory for model download", path=target, free_gb=round(best_free / (1024**3), 1))
    return target


def download_gcs_prefix(gcs_uri: str, local_dir: str) -> tuple[str, bool]:
    """Downloads a gs://bucket/prefix directory to local_dir using parallel HTTPS streams."""
    if not gcs_uri.startswith("gs://"):
        return gcs_uri, False

    # Fast path if Cloud Storage FUSE is mounted at /gcs
    fuse_path = "/gcs/" + gcs_uri[len("gs://") :]
    if os.path.isdir(fuse_path) and os.path.exists(os.path.join(fuse_path, "config.json")):
        log("using /gcs FUSE mount directly", path=fuse_path)
        return fuse_path, False

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    if os.path.exists(os.path.join(local_dir, "config.json")):
        return local_dir, True

    without_scheme = gcs_uri[len("gs://") :].rstrip("/")
    bucket, _, prefix = without_scheme.partition("/")
    prefix_with_slash = prefix + "/" if prefix else ""

    token = get_gcp_token()
    objects = []
    page_token = None
    while True:
        q = {"prefix": prefix_with_slash}
        if page_token:
            q["pageToken"] = page_token
        list_url = f"https://storage.googleapis.com/storage/v1/b/{urllib.parse.quote(bucket)}/o?{urllib.parse.urlencode(q)}"
        req = urllib.request.Request(list_url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for item in data.get("items", []):
            name = item["name"]
            if not name.endswith("/"):
                objects.append(name)
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    log("downloading model artifacts from GCS", bucket=bucket, prefix=prefix_with_slash, files=len(objects))
    t0 = time.time()

    def fetch_one(obj_name: str) -> None:
        rel = obj_name[len(prefix_with_slash) :] if obj_name.startswith(prefix_with_slash) else os.path.basename(obj_name)
        if "/" in rel:
            return
        dest = os.path.join(local_dir, rel)
        dl_url = f"https://storage.googleapis.com/storage/v1/b/{urllib.parse.quote(bucket)}/o/{urllib.parse.quote(obj_name, safe='')}?alt=media"
        dl_req = urllib.request.Request(dl_url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(dl_req, timeout=300) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, length=16 * 1024 * 1024)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch_one, objects))

    log("completed GCS model download", seconds=round(time.time() - t0, 1), local_dir=local_dir)
    return local_dir, True


def load_model_background() -> None:
    global MODEL, TOKENIZER, LOAD_ERROR
    try:
        storage_uri = os.environ.get("AIP_STORAGE_URI") or os.environ.get("MODEL_ARTIFACT_GCS_URI") or "google/gemma-4-12B-it"
        local_model_dir, is_temp_copy = download_gcs_prefix(storage_uri, pick_scratch_dir())

        log("loading tokenizer", source=local_model_dir)
        tok = AutoTokenizer.from_pretrained(local_model_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        common_kwargs = dict(
            device_map={"": 0} if torch.cuda.is_available() else "auto",
            dtype=torch.bfloat16,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
        )
        log("loading fine-tuned Gemma 4 model onto GPU in 4-bit NF4", source=local_model_dir)
        t0 = time.time()
        try:
            from transformers import AutoModelForMultimodalLM
            model = AutoModelForMultimodalLM.from_pretrained(
                local_model_dir,
                attn_implementation="sdpa",
                **common_kwargs,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                local_model_dir,
                attn_implementation="eager",
                **common_kwargs,
            )
        model.eval()
        TOKENIZER = tok
        MODEL = model
        if is_temp_copy:
            shutil.rmtree(local_model_dir, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        MODEL_READY_EVENT.set()
        log("model ready for inference", load_seconds=round(time.time() - t0, 1))
    except Exception as e:
        LOAD_ERROR = str(e)
        MODEL_READY_EVENT.set()
        log("FATAL model load error", error=LOAD_ERROR)


def generate_text(prompt: str = "", messages=None, max_tokens: int = 512, temperature: float = 0.1, top_p: float = 0.95) -> str:
    with GENERATE_LOCK:
        if messages:
            try:
                prompt = TOKENIZER.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
                )
            except TypeError:
                prompt = TOKENIZER.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
        inputs = TOKENIZER(prompt, add_special_tokens=False, return_tensors="pt")
        inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=int(max_tokens),
            pad_token_id=TOKENIZER.pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
        )
        if float(temperature) > 0.01:
            gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=float(top_p))
        else:
            gen_kwargs.update(do_sample=False)

        with torch.inference_mode():
            output_ids = MODEL.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0, input_len:]
        return TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()


class PredictionHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/health") or self.path.startswith("/ping"):
            if LOAD_ERROR:
                self._send_json(500, {"status": "error", "error": LOAD_ERROR})
            else:
                # Return 200 OK while background thread loads weights so Vertex AI liveness probe never restarts container
                self._send_json(200, {"status": "healthy", "model_loaded": MODEL_READY_EVENT.is_set()})
        else:
            self._send_json(200, {"status": "ok", "model_ready": MODEL_READY_EVENT.is_set()})

    def do_POST(self) -> None:
        MODEL_READY_EVENT.wait(timeout=600)
        if LOAD_ERROR or MODEL is None:
            self._send_json(503, {"error": LOAD_ERROR or "Model is still loading"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        data = json.loads(raw)

        # 1. Vertex AI Prediction format: {"instances": [{"prompt": "...", "max_tokens": 512, ...}]}
        if "instances" in data:
            predictions = []
            for inst in data["instances"]:
                if isinstance(inst, str):
                    predictions.append(generate_text(prompt=inst))
                else:
                    predictions.append(
                        generate_text(
                            prompt=inst.get("prompt", ""),
                            messages=inst.get("messages"),
                            max_tokens=inst.get("max_tokens", 512),
                            temperature=inst.get("temperature", 0.1),
                            top_p=inst.get("top_p", 0.95),
                        )
                    )
            self._send_json(200, {"predictions": predictions})
            return

        # 2. OpenAI /v1/chat/completions format
        if "messages" in data:
            text = generate_text(
                messages=data["messages"],
                max_tokens=data.get("max_tokens", 512),
                temperature=data.get("temperature", 0.1),
                top_p=data.get("top_p", 0.95),
            )
            self._send_json(
                200,
                {
                    "id": "chatcmpl-fsi",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                },
            )
            return

        # 3. OpenAI /v1/completions format
        text = generate_text(
            prompt=data.get("prompt", ""),
            max_tokens=data.get("max_tokens", 512),
            temperature=data.get("temperature", 0.1),
            top_p=data.get("top_p", 0.95),
        )
        self._send_json(
            200,
            {
                "id": "cmpl-fsi",
                "object": "text_completion",
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            },
        )


def main() -> None:
    port = int(os.environ.get("AIP_HTTP_PORT", os.environ.get("PORT", "8080")))
    threading.Thread(target=load_model_background, daemon=True).start()
    log("starting Sovereign Gemma HTTP inference server", port=port)
    server = ThreadingHTTPServer(("0.0.0.0", port), PredictionHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
