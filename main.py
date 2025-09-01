from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from typing import List
from starlette.concurrency import run_in_threadpool

translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]     # e.g. { "texts": ["Hello", "How are you?"] }

@app.post("/translate")
def translate(req: TranslationRequest):
    result = translator(req.text, max_length=512)
    return {"translation": result[0]['translation_text']}

@app.post("/translate_batch")
async def translate_batch(req: BatchRequest):
    texts = req.texts
    # run heavy CPU work in threadpool so uvicorn event loop isn't blocked
    def do_translate(batch):
        # The pipeline accepts a list and will batch internally
        results = translator(batch, max_length=512)
        # results is a list of dicts like [{"translation_text": "..."}]
        return [r["translation_text"] for r in results]

    translations = await run_in_threadpool(do_translate, texts)
    return {"translations": translations}