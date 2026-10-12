# Sovereign Gemma QLoRA Fine-Tuning Container
# Built remotely in Google Cloud Build (asia-south1) — NO local Docker daemon required!
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

WORKDIR /workspace

# Install Google Cloud SDK for seamless gs:// bucket I/O inside air-gapped VPC-SC
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    git \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update -y && apt-get install google-cloud-cli -y \
    && rm -rf /var/lib/apt/lists/*

# Install GPU Fine-Tuning Python Dependencies
COPY requirements-train.txt /workspace/requirements-train.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements-train.txt

# Copy training script
COPY train_gemma_qlora.py /workspace/train_gemma_qlora.py

ENTRYPOINT ["python3", "/workspace/train_gemma_qlora.py"]
