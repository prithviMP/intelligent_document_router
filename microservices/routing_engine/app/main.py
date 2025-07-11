from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Routing Engine Service")

class RouteRequest(BaseModel):
    document_id: str
    doc_type: Optional[str] = None

class RouteResponse(BaseModel):
    assigned_to: str
    priority: int

@app.get("/ping")
def ping():
    return {"message": "pong from Routing Engine"}

@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest = Body(...)):
    # Dummy logic for now
    return {"assigned_to": "legal_team", "priority": 1} 