import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, RobertaForTokenClassification
from infer import predict_ner, tokenize_vietnamese

# --- Pydantic Models ---
class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    tokens: List[str]
    tags: List[str]
    confidence_scores: List[float]

# --- FastAPI App ---
app = FastAPI(
    title="Vietnamese NER API",
    description="An API for Named Entity Recognition in Vietnamese text.",
    version="1.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Model Loading ---
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/rb071/.phoner_covid19/models/hf//sroie2019v1")
tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        model = RobertaForTokenClassification.from_pretrained(MODEL_PATH)
        print(f"Successfully loaded model and tokenizer from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Depending on the desired behavior, you might want to raise the exception
        # or handle it gracefully.

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vietnamese NER API"}


@app.post("/predict", response_model=NERResponse)
def predict(request: NERRequest):
    """
    Nhận văn bản đầu vào và trả về kết quả dự đoán NER.
    """
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    try:
        tokens, tags, scores = predict_ner(MODEL_PATH, request.text, tokenizer, model)
        return NERResponse(tokens=tokens, tags=tags, confidence_scores=scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


