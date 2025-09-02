# syntax=docker/dockerfile:1

########################
# Stage 1: Build & convert
########################
FROM python:3.11-bullseye AS builder

# System deps for building + conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies (including torch for conversion)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Convert the HF model into CTranslate2 format
RUN ct2-transformers-converter \
      --model Helsinki-NLP/opus-mt-en-fr \
      --output_dir /app/ct2_enfr \
      --quantization int8 \
      --force

########################
# Stage 2: Runtime only
########################
FROM python:3.11-bullseye

WORKDIR /app

# Only install what’s needed for serving
# (drop torch since inference is on ctranslate2)
COPY requirements.txt /app/
RUN pip install --no-cache-dir \
      fastapi==0.100.0 \
      uvicorn==0.22.0 \
      transformers==4.31.0 \
      sentencepiece==0.1.99 \
      ctranslate2==3.24.0 \
      huggingface_hub>=0.23.0 \
      "numpy<2"

# Copy converted model + app code from builder
COPY --from=builder /app/ct2_enfr /app/ct2_enfr
COPY main.py /app/main.py

# Env vars
ENV CT2_MODEL_DIR=/app/ct2_enfr
ENV PORT=8000

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]