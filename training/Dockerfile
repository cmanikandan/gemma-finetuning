# Sovereign Gemma 4 QLoRA Fine-Tuning Container
# Built remotely in Google Cloud Build (asia-south1) — NO local Docker daemon required!
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

# Install GPU Fine-Tuning Python Dependencies (Base image already provides torch 2.13 + CUDA 12.6)
COPY requirements-train.txt /workspace/requirements-train.txt
RUN pip install -r /workspace/requirements-train.txt && \
    python -c "import torch, transformers, trl, peft, bitsandbytes; print('[+] Verified Gemma 4 image stack:', torch.__version__, transformers.__version__, trl.__version__, peft.__version__)"

# Copy training script
COPY train_gemma_qlora.py /workspace/train_gemma_qlora.py

ENTRYPOINT ["python", "/workspace/train_gemma_qlora.py"]
