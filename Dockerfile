# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System packages (git to fetch model from HF hub; build-essential sometimes helps wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Convert the HF model to CTranslate2 format (int8 quantization)
# This runs at build time so the container starts fast later.
# We also copy tokenizer files so runtime can detokenize.
RUN ct2-transformers-converter \
      --model Helsinki-NLP/opus-mt-en-fr \
      --output_dir /app/ct2_enfr \
      --quantization int8 \
      --force

# App code last (so editing app doesn't invalidate earlier layers)
COPY main.py /app/main.py

# Expose a predictable env var for the model path
ENV CT2_MODEL_DIR=/app/ct2_enfr
ENV PORT=8000

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]