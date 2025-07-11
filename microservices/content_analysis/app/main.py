from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Content Analysis Service")

class AnalyzeRequest(BaseModel):
    document_id: str
    content: Optional[str] = None

class AnalyzeResponse(BaseModel):
    key_entities: List[str]
    summary: str
    sentiment: str

@app.get("/ping")
def ping():
    return {"message": "pong from Content Analysis Service"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest = Body(...)):
    # Dummy logic for now
    return {
        "key_entities": ["Alice", "Bob", "Acme Corp"],
        "summary": "This is a dummy summary.",
        "sentiment": "neutral"
    } 