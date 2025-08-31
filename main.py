from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# 1) Load model (English -> French)
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

# 2) FastAPI app
app = FastAPI(title="Helsinki EN->FR Translator")

# 3) CORS (dev-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in prod, restrict to your frontend domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4) Request schema
class TranslationRequest(BaseModel):
    text: str

# 5) Endpoints
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/translate")
def translate(req: TranslationRequest):
    result = translator(req.text, max_length=512)
    return {"translation": result[0]["translation_text"]}
