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


import sys
import os
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form, Query
from sqlalchemy.orm import Session
from database import SessionLocal, Document
from classification.classifier import classify_document, extract_text_from_file
import classification.departments as dept_module
from datetime import datetime
from models import BulkDocumentResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = FastAPI()

ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'docx', 'doc', 'xlsx', 'xls', 
    'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'csv'
}

def is_allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not is_allowed_file(file.filename):
            raise HTTPException(status_code=400, detail=f"File type not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")

        content = await file.read()
        text = extract_text_from_file(content, file.filename)
        classification = classify_document(text, file.filename)

        # Save to database
        db: Session = SessionLocal()

        # Extract departments and IDs from classification
        departments = classification.get("departments", ["General"])
        department_ids = classification.get("department_ids", [9])
        department = ", ".join(departments)
        department_id = department_ids[0] if department_ids else 9  # Use first department ID

        doc = Document(
            filename=file.filename,
            department=department,
            department_id=department_id,
            extracted_date=classification["date"],
            priority=classification["priority"]
        )

        db.add(doc)
        db.commit()
        db.refresh(doc)
        db.close()

        return {
            "filename": file.filename,
            "department": department,
            "department_id": department_id,
            "priority": classification["priority"],
            "extracted_text_length": len(text)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/documents/")
def get_all_documents():
    db: Session = SessionLocal()
    docs = db.query(Document).all()
    db.close()
    return [{
        "id": doc.id,
        "filename": doc.filename,
        "department": doc.department,
        "department_id": doc.department_id,
        "priority": doc.priority,
        "upload_time": doc.upload_time
    } for doc in docs]

@app.get("/department/")
def get_departments(department_id: Optional[int] = Query(None)):
    if department_id:
        dept = dept_module.get_department_by_id(department_id)
        if dept:
            return {
                "id": department_id,
                "name": dept["name"],
                "keywords": dept["keywords"]
            }
        else:
            raise HTTPException(status_code=404, detail="Department not found")
    else:
        return {
            "departments": [
                {
                    "id": dept_id,
                    "name": dept_info["name"],
                    "keywords": dept_info["keywords"]
                }
                for dept_id, dept_info in dept_module.DEPARTMENTS.items()
            ]
        }

@app.post("/department/")
def add_department(name: str = Body(..., embed=True),
                   keywords: List[str] = Body(..., embed=True)):
    name = name.lower()
    new_keywords = [kw.lower() for kw in keywords]

    # Find next available ID
    next_id = max(dept_module.DEPARTMENTS.keys()) + 1 if dept_module.DEPARTMENTS else 1

    # Check if department already exists
    existing_id = dept_module.get_department_id_by_name(name)
    if existing_id:
        # Update existing department
        dept_module.DEPARTMENTS[existing_id]["keywords"].extend(new_keywords)
        # Remove duplicates
        dept_module.DEPARTMENTS[existing_id]["keywords"] = list(set(dept_module.DEPARTMENTS[existing_id]["keywords"]))
        message = f"Updated department '{name}' with new keywords."
        return {
            "message": message,
            "department_id": existing_id,
            "keywords": dept_module.DEPARTMENTS[existing_id]["keywords"]
        }
    else:
        # Add new department
        dept_module.DEPARTMENTS[next_id] = {
            "name": name,
            "keywords": new_keywords
        }
        message = f"Added new department '{name}' with ID {next_id}."
        return {
            "message": message,
            "department_id": next_id,
            "keywords": new_keywords
        }

@app.post("/upload/bulk/")
async def bulk_upload(files: List[UploadFile] = File(...)):
    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="You can upload a maximum of 20 files at once."
        )

    db: Session = SessionLocal()
    responses = []

    for index, file in enumerate(files):
        try:
            if not is_allowed_file(file.filename):
                responses.append({
                    "index": index,
                    "filename": file.filename,
                    "error": f"File type not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                })
                continue

            content = await file.read()
            text = extract_text_from_file(content, file.filename)
            classification = classify_document(text, file.filename)

            # Extract departments and IDs from classification
            departments = classification.get("departments", ["General"])
            department_ids = classification.get("department_ids", [9])
            department = ", ".join(departments)
            department_id = department_ids[0] if department_ids else 9

            doc = Document(
                filename=file.filename,
                department=department,
                department_id=department_id,
                extracted_date=classification["date"],
                priority=classification["priority"]
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            responses.append({
                "index": index,
                "filename": file.filename,
                "department": department,
                "department_id": department_id,
                "priority": classification["priority"],
                "extracted_text_length": len(text)
            })

        except Exception as e:
            responses.append({
                "index": index,
                "filename": file.filename,
                "error": str(e)
            })

    db.close()
    return {"documents": responses}

@app.get("/supported-formats/")
def get_supported_formats():
    return {
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "description": {
            "documents": ["pdf", "docx", "doc", "txt"],
            "spreadsheets": ["xlsx", "xls", "csv"],
            "images": ["jpg", "jpeg", "png", "bmp", "tiff", "gif"]
        }
    }
