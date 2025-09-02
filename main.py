import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import ctranslate2
from transformers import AutoTokenizer
from starlette.concurrency import run_in_threadpool

# -----------------------------------------------------------------------------
# 1) Load tokenizer + CT2 model
# -----------------------------------------------------------------------------
# We keep tokenizer from HF to handle SentencePiece (Marian uses it)
TOKENIZER_ID = "Helsinki-NLP/opus-mt-en-fr"
CT2_MODEL_DIR = os.environ.get("CT2_MODEL_DIR", "ct2_enfr")  # set in Dockerfile

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

# For CPU: compute_type="int8" (if you converted with int8 quantization)
# Tune threads: intra_threads = number of CPU cores for compute
translator_ct2 = ctranslate2.Translator(
    CT2_MODEL_DIR,
    device="cpu",
    compute_type="int8",
    # You can tune these depending on the machine:
    intra_threads=int(os.environ.get("CT2_INTRA_THREADS", "4")),
    inter_threads=int(os.environ.get("CT2_INTER_THREADS", "1")),
)

# -----------------------------------------------------------------------------
# 2) FastAPI app + CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Helsinki EN->FR Translator (CTranslate2)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# 3) Schemas
# -----------------------------------------------------------------------------
class TranslationRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

# -----------------------------------------------------------------------------
# 4) Helpers: tokenize → translate → detokenize
# -----------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    # We want tokens (string pieces), not ids. Use add_special_tokens=True so BOS/EOS are handled.
    ids = tokenizer.encode(text, add_special_tokens=True)
    return tokenizer.convert_ids_to_tokens(ids)

def _detokenize(tokens: List[str]) -> str:
    ids = tokenizer.convert_tokens_to_ids(tokens)
    # skip_special_tokens=True cleans up <s>, </s>, etc
    return tokenizer.decode(ids, skip_special_tokens=True)

def translate_texts_ct2(texts: List[str], beam_size: int = 1, max_len: int = 256) -> List[str]:
    # Tokenize each input string to SentencePiece tokens
    src_tokens = [_tokenize(t) for t in texts]
    # Run fast translation
    results = translator_ct2.translate_batch(
        src_tokens,
        beam_size=beam_size,
        max_decoding_length=max_len,
    )
    # Detokenize the best hypothesis for each input
    outputs = [_detokenize(r.hypotheses[0]) for r in results]
    return outputs

# -----------------------------------------------------------------------------
# 5) Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/translate")
def translate(req: TranslationRequest):
    out = translate_texts_ct2([req.text])[0]
    return {"translation": out}

@app.post("/translate_batch")
async def translate_batch(req: BatchRequest):
    # Offload CPU work to a threadpool so the event loop stays responsive
    def work():
        return translate_texts_ct2(req.texts)

    translations = await run_in_threadpool(work)
    return {"translations": translations}
