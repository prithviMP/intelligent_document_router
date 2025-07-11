"""
Documents router for document upload, classification, and management
"""

import os
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import httpx
import asyncio

from libs.database.connection import get_db
from libs.database.models import Document, User, Metadata
from app.dependencies import get_current_user

router = APIRouter()

# Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
ALLOWED_FILE_TYPES = os.getenv("ALLOWED_FILE_TYPES", "pdf,doc,docx,txt,jpg,jpeg,png").split(",")
CLASSIFICATION_SERVICE_URL = os.getenv("CLASSIFICATION_SERVICE_URL", "http://classification:8001")

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and classify a document"""
    
    # Validate file size
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE} bytes"
        )
    
    # Validate file type
    file_extension = file.filename.split('.')[-1].lower() if file.filename else ''
    if file_extension not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_FILE_TYPES)}"
        )
    
    try:
        # Create document record
        doc_id = uuid.uuid4()
        storage_path = f"documents/{doc_id}/{file.filename}"
        
        document = Document(
            id=doc_id,
            original_name=file.filename,
            storage_path=storage_path,
            file_size=file.size,
            mime_type=file.content_type,
            status="uploaded"
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Send to classification service
        try:
            async with httpx.AsyncClient() as client:
                # For now, just return success
                # In full implementation, this would send the file to classification service
                response = await client.post(
                    f"{CLASSIFICATION_SERVICE_URL}/classify",
                    files={"file": (file.filename, file.file, file.content_type)},
                    data={"doc_id": str(doc_id)}
                )
                
                if response.status_code == 200:
                    classification_result = response.json()
                    # Update document with classification results
                    document.doc_type = classification_result.get("doc_type")
                    document.confidence = classification_result.get("confidence", 0.0)
                    document.status = "classified"
                    db.commit()
                    
                    return {
                        "message": "Document uploaded and classified successfully",
                        "doc_id": str(doc_id),
                        "classification": classification_result
                    }
                else:
                    # Fallback: mark as pending classification
                    document.status = "pending_classification"
                    db.commit()
                    
                    return {
                        "message": "Document uploaded successfully, classification pending",
                        "doc_id": str(doc_id),
                        "status": "pending_classification"
                    }
                    
        except Exception as e:
            # Fallback: mark as pending classification
            document.status = "pending_classification"
            db.commit()
            
            return {
                "message": "Document uploaded successfully, classification pending",
                "doc_id": str(doc_id),
                "status": "pending_classification",
                "error": str(e)
            }
            
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )

@router.get("/")
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    doc_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List documents with optional filtering"""
    
    query = db.query(Document)
    
    # Apply filters
    if doc_type:
        query = query.filter(Document.doc_type == doc_type)
    if status:
        query = query.filter(Document.status == status)
    
    # Get total count
    total_count = query.count()
    
    # Apply pagination
    documents = query.offset(skip).limit(limit).all()
    
    return {
        "documents": [
            {
                "id": str(doc.id),
                "original_name": doc.original_name,
                "doc_type": doc.doc_type,
                "confidence": doc.confidence,
                "status": doc.status,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "processed_at": doc.processed_at.isoformat() if doc.processed_at else None
            }
            for doc in documents
        ],
        "total_count": total_count,
        "skip": skip,
        "limit": limit
    }

@router.get("/{doc_id}")
async def get_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get document details by ID"""
    
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )
    
    document = db.query(Document).filter(Document.id == doc_uuid).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Get metadata
    metadata = db.query(Metadata).filter(Metadata.doc_id == doc_uuid).first()
    
    return {
        "id": str(document.id),
        "original_name": document.original_name,
        "doc_type": document.doc_type,
        "confidence": document.confidence,
        "status": document.status,
        "file_size": document.file_size,
        "mime_type": document.mime_type,
        "created_at": document.created_at.isoformat() if document.created_at else None,
        "processed_at": document.processed_at.isoformat() if document.processed_at else None,
        "metadata": {
            "key_entities": metadata.key_entities if metadata else None,
            "key_phrases": metadata.key_phrases if metadata else None,
            "summary": metadata.summary if metadata else None,
            "risk_score": metadata.risk_score if metadata else None,
            "sentiment_score": metadata.sentiment_score if metadata else None,
            "language": metadata.language if metadata else None
        } if metadata else None
    }

@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a document"""
    
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )
    
    document = db.query(Document).filter(Document.id == doc_uuid).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        db.delete(document)
        db.commit()
        return {"message": "Document deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.post("/{doc_id}/reclassify")
async def reclassify_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Reclassify a document"""
    
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )
    
    document = db.query(Document).filter(Document.id == doc_uuid).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Update status to trigger reclassification
    document.status = "pending_classification"
    db.commit()
    
    return {
        "message": "Document queued for reclassification",
        "doc_id": str(doc_id)
    } 