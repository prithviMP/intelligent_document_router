from pydantic import BaseModel
from datetime import date
from typing import List

class DocumentCreate(BaseModel):
    filename: str
    content: str
    extracted_date: date

class BulkDocumentResponse(BaseModel):
    documents: List[DocumentCreate]
