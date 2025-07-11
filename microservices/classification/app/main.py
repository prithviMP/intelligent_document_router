from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Classification Service")

class ClassifyRequest(BaseModel):
    document_id: str
    content: Optional[str] = None

class ClassifyResponse(BaseModel):
    doc_type: str
    confidence: float

@app.get("/ping")
def ping():
    return {"message": "pong from Classification Service"}

@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest = Body(...)):
    # Dummy logic for now
    return {"doc_type": "contract", "confidence": 0.95} 