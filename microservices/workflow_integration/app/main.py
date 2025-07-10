from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Workflow Integration Service")

class NotifyRequest(BaseModel):
    recipient: str
    message: str
    channel: Optional[str] = None

class NotifyResponse(BaseModel):
    status: str
    detail: str

@app.get("/ping")
def ping():
    return {"message": "pong from Workflow Integration Service"}

@app.post("/notify", response_model=NotifyResponse)
def notify(request: NotifyRequest = Body(...)):
    # Dummy logic for now
    return {"status": "sent", "detail": f"Notification sent to {request.recipient}"} 