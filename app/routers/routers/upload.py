from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, Document
from classification.classifier import classify_document
from shared.priority import calculate_priority
from datetime import datetime
import os

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload/")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Read file contents
    contents = await file.read()
    text = contents.decode("utf-8", errors="ignore")
    filename = str(file.filename) if file.filename else "unknown"
    filepath = os.path.join(UPLOAD_DIR, filename)

    # Save file
    with open(filepath, "wb") as f:
        f.write(contents)

    # Classify and extract metadata
    classification = classify_document(text)
    departments = classification.get("departments", ["General"])
    department = ", ".join(departments)
    extracted_date = classification.get("date", datetime.today().date())
    priority = classification.get("priority", "Normal")

    # Create DB record
    doc = Document(
        filename=filename,
        department=department,
        extracted_date=extracted_date,
        priority=priority
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {
        "filename": filename,
        "department": department,
        "priority": priority
    }
