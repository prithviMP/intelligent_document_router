
"""
Content Analysis Service - Main application
Handles document content extraction, entity recognition, and analysis
"""

import os
import sys
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import json
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import httpx
import aio_pika
import asyncpg

# Document processing libraries
import PyMuPDF as fitz  # PDF processing
import pdfplumber
import PyPDF2
from docx import Document as DocxDocument
import pytesseract
from PIL import Image
import io
import spacy
from spacy import displacy
import re
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import hashlib

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/content_analysis")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Content Analysis Service",
    description="AI-powered document content analysis and extraction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Models
class ContentAnalysisJob(Base):
    __tablename__ = 'content_analysis_jobs'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_size = Column(Float)
    mime_type = Column(String(100))

    # Extracted content
    extracted_text = Column(Text)
    text_hash = Column(String(64))  # SHA256 hash of text

    # Entity extraction
    entities = Column(JSON)  # Names, organizations, dates, etc.
    key_phrases = Column(JSON)
    topics = Column(JSON)

    # Analysis results
    summary = Column(Text)
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    language = Column(String(10))

    # Risk and compliance
    risk_score = Column(Float)
    compliance_flags = Column(JSON)
    sensitive_data_detected = Column(Boolean, default=False)

    # Metadata
    word_count = Column(Float)
    page_count = Column(Float)
    processing_time = Column(Float)
    confidence_score = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed

class DocumentRelationship(Base):
    __tablename__ = 'document_relationships'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_document_id = Column(UUID(as_uuid=True), nullable=False)
    target_document_id = Column(UUID(as_uuid=True), nullable=False)
    relationship_type = Column(String(50))  # similar, references, follows_up, etc.
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class AnalyzeRequest(BaseModel):
    document_id: str
    filename: str
    content_type: Optional[str] = None
    analyze_sentiment: bool = True
    extract_entities: bool = True
    generate_summary: bool = True
    detect_compliance_issues: bool = True

class EntityInfo(BaseModel):
    text: str
    label: str
    confidence: float
    start_pos: int
    end_pos: int

class AnalyzeResponse(BaseModel):
    document_id: str
    status: str
    extracted_text: Optional[str] = None
    entities: List[EntityInfo] = []
    key_phrases: List[str] = []
    topics: List[str] = []
    summary: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    language: Optional[str] = None
    risk_score: Optional[float] = None
    compliance_flags: List[str] = []
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    processing_time: Optional[float] = None
    confidence_score: Optional[float] = None

class BulkAnalyzeRequest(BaseModel):
    documents: List[AnalyzeRequest]

# Document processing functions
class DocumentProcessor:
    def __init__(self):
        self.nlp = nlp
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def extract_text_from_pdf(self, file_content: bytes, method: str = "pymupdf") -> str:
        """Extract text from PDF using multiple methods"""
        text = ""

        if method == "pymupdf":
            try:
                doc = fitz.open(stream=file_content, filetype="pdf")
                for page in doc:
                    text += page.get_text()
                doc.close()
            except Exception as e:
                logger.error(f"PyMuPDF extraction failed: {e}")
                method = "pdfplumber"

        if method == "pdfplumber" or not text.strip():
            try:
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                logger.error(f"pdfplumber extraction failed: {e}")
                method = "pypdf2"

        if method == "pypdf2" or not text.strip():
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e:
                logger.error(f"PyPDF2 extraction failed: {e}")

        return text.strip()

    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX files"""
        try:
            doc = DocxDocument(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""

    def extract_text_from_image(self, file_content: bytes) -> str:
        """Extract text from images using OCR"""
        try:
            image = Image.open(io.BytesIO(file_content))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""

    def extract_text_from_file(self, file_content: bytes, filename: str, mime_type: str = None) -> str:
        """Extract text from various file formats"""
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''

        if file_ext == 'pdf' or (mime_type and 'pdf' in mime_type):
            return self.extract_text_from_pdf(file_content)
        elif file_ext in ['docx', 'doc'] or (mime_type and 'word' in mime_type):
            return self.extract_text_from_docx(file_content)
        elif file_ext in ['txt', 'md', 'py', 'js', 'html', 'css']:
            return file_content.decode('utf-8', errors='ignore')
        elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'] or (mime_type and 'image' in mime_type):
            return self.extract_text_from_image(file_content)
        else:
            # Try to decode as text
            try:
                return file_content.decode('utf-8', errors='ignore')
            except:
                return ""

    def extract_entities(self, text: str) -> List[EntityInfo]:
        """Extract named entities from text"""
        entities = []

        if not self.nlp:
            return entities

        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append(EntityInfo(
                    text=ent.text,
                    label=ent.label_,
                    confidence=0.85,  # spaCy doesn't provide confidence scores
                    start_pos=ent.start_char,
                    end_pos=ent.end_char
                ))
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")

        return entities

    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text"""
        try:
            # Tokenize and clean text
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in self.stop_words]

            # Get lemmatized words
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]

            # Count word frequencies
            word_freq = Counter(lemmatized_words)

            # Extract n-grams
            phrases = []

            # Bigrams
            for i in range(len(words) - 1):
                if words[i] not in self.stop_words and words[i+1] not in self.stop_words:
                    phrases.append(f"{words[i]} {words[i+1]}")

            # Trigrams
            for i in range(len(words) - 2):
                if all(word not in self.stop_words for word in words[i:i+3]):
                    phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")

            # Combine single words and phrases
            all_phrases = list(word_freq.keys()) + phrases
            phrase_freq = Counter(all_phrases)

            return [phrase for phrase, _ in phrase_freq.most_common(max_phrases)]

        except Exception as e:
            logger.error(f"Key phrase extraction failed: {e}")
            return []

    def analyze_sentiment(self, text: str) -> tuple[float, str]:
        """Analyze sentiment of text"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"

            return polarity, label
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0, "neutral"

    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate text summary"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= max_sentences:
                return text

            # Simple extractive summarization
            # Calculate sentence scores based on word frequency
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in self.stop_words]
            word_freq = Counter(words)

            sentence_scores = {}
            for sentence in sentences:
                sentence_words = word_tokenize(sentence.lower())
                sentence_words = [word for word in sentence_words if word.isalpha()]

                if sentence_words:
                    sentence_scores[sentence] = sum(word_freq.get(word, 0) for word in sentence_words) / len(sentence_words)

            # Get top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]

            # Sort by original order
            summary_sentences = []
            for sentence in sentences:
                if any(sentence == s[0] for s in top_sentences):
                    summary_sentences.append(sentence)
                if len(summary_sentences) >= max_sentences:
                    break

            return " ".join(summary_sentences)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return text[:500] + "..." if len(text) > 500 else text

    def detect_compliance_issues(self, text: str) -> tuple[float, List[str]]:
        """Detect compliance and sensitive data issues"""
        flags = []
        risk_score = 0.0

        try:
            # Check for sensitive data patterns
            patterns = {
                'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'phone': r'\b\d{3}-\d{3}-\d{4}\b',
                'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            }

            for pattern_name, pattern in patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    flags.append(f"Contains {pattern_name}")
                    risk_score += 0.2

            # Check for confidential keywords
            confidential_keywords = [
                'confidential', 'proprietary', 'classified', 'restricted',
                'internal only', 'do not distribute', 'sensitive',
                'personal information', 'private', 'secret'
            ]

            text_lower = text.lower()
            for keyword in confidential_keywords:
                if keyword in text_lower:
                    flags.append(f"Contains confidential keyword: {keyword}")
                    risk_score += 0.1

            # Normalize risk score
            risk_score = min(risk_score, 1.0)

        except Exception as e:
            logger.error(f"Compliance detection failed: {e}")

        return risk_score, flags

    def detect_language(self, text: str) -> str:
        """Detect text language"""
        try:
            blob = TextBlob(text)
            return blob.detect_language()
        except:
            return "en"  # Default to English

# Initialize processor
processor = DocumentProcessor()

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "content_analysis",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(
    file: UploadFile = File(...),
    analyze_sentiment: bool = True,
    extract_entities: bool = True,
    generate_summary: bool = True,
    detect_compliance_issues: bool = True,
    db: Session = Depends(get_db)
):
    """Analyze uploaded document"""
    start_time = datetime.utcnow()
    document_id = str(uuid.uuid4())

    try:
        # Read file content
        file_content = await file.read()

        # Create job record
        job = ContentAnalysisJob(
            document_id=document_id,
            filename=file.filename,
            file_size=len(file_content),
            mime_type=file.content_type,
            status="processing"
        )
        db.add(job)
        db.commit()

        # Extract text
        extracted_text = processor.extract_text_from_file(
            file_content, file.filename, file.content_type
        )

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from document")

        # Calculate text hash
        text_hash = hashlib.sha256(extracted_text.encode()).hexdigest()

        # Initialize results
        entities = []
        key_phrases = []
        summary = None
        sentiment_score = None
        sentiment_label = None
        risk_score = 0.0
        compliance_flags = []

        # Extract entities
        if extract_entities:
            entities = processor.extract_entities(extracted_text)

        # Extract key phrases
        key_phrases = processor.extract_key_phrases(extracted_text)

        # Generate summary
        if generate_summary:
            summary = processor.generate_summary(extracted_text)

        # Analyze sentiment
        if analyze_sentiment:
            sentiment_score, sentiment_label = processor.analyze_sentiment(extracted_text)

        # Detect compliance issues
        if detect_compliance_issues:
            risk_score, compliance_flags = processor.detect_compliance_issues(extracted_text)

        # Detect language
        language = processor.detect_language(extracted_text)

        # Calculate metrics
        word_count = len(extracted_text.split())
        page_count = extracted_text.count('\f') + 1  # Form feed character indicates page break
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        confidence_score = 0.9 if entities else 0.7  # Simple confidence calculation

        # Update job record
        job.extracted_text = extracted_text
        job.text_hash = text_hash
        job.entities = [entity.dict() for entity in entities]
        job.key_phrases = key_phrases
        job.summary = summary
        job.sentiment_score = sentiment_score
        job.sentiment_label = sentiment_label
        job.language = language
        job.risk_score = risk_score
        job.compliance_flags = compliance_flags
        job.sensitive_data_detected = risk_score > 0.3
        job.word_count = word_count
        job.page_count = page_count
        job.processing_time = processing_time
        job.confidence_score = confidence_score
        job.status = "completed"

        db.commit()

        return AnalyzeResponse(
            document_id=document_id,
            status="completed",
            extracted_text=extracted_text,
            entities=entities,
            key_phrases=key_phrases,
            summary=summary,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            language=language,
            risk_score=risk_score,
            compliance_flags=compliance_flags,
            word_count=word_count,
            page_count=page_count,
            processing_time=processing_time,
            confidence_score=confidence_score
        )

    except Exception as e:
        # Update job status to failed
        if 'job' in locals():
            job.status = "failed"
            db.commit()

        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/bulk")
async def analyze_documents_bulk(
    files: List[UploadFile] = File(...),
    analyze_sentiment: bool = True,
    extract_entities: bool = True,
    generate_summary: bool = True,
    detect_compliance_issues: bool = True,
    db: Session = Depends(get_db)
):
    """Analyze multiple documents"""
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files allowed")

    results = []

    for file in files:
        try:
            result = await analyze_document(
                file=file,
                analyze_sentiment=analyze_sentiment,
                extract_entities=extract_entities,
                generate_summary=generate_summary,
                detect_compliance_issues=detect_compliance_issues,
                db=db
            )
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })

    return {"results": results}

@app.get("/analyze/{document_id}")
async def get_analysis(document_id: str, db: Session = Depends(get_db)):
    """Get analysis results for a document"""
    job = db.query(ContentAnalysisJob).filter(
        ContentAnalysisJob.document_id == document_id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Document not found")

    entities = [EntityInfo(**entity) for entity in job.entities] if job.entities else []

    return AnalyzeResponse(
        document_id=str(job.document_id),
        status=job.status,
        extracted_text=job.extracted_text,
        entities=entities,
        key_phrases=job.key_phrases or [],
        summary=job.summary,
        sentiment_score=job.sentiment_score,
        sentiment_label=job.sentiment_label,
        language=job.language,
        risk_score=job.risk_score,
        compliance_flags=job.compliance_flags or [],
        word_count=int(job.word_count) if job.word_count else None,
        page_count=int(job.page_count) if job.page_count else None,
        processing_time=job.processing_time,
        confidence_score=job.confidence_score
    )

@app.get("/documents/{document_id}/relationships")
async def get_document_relationships(document_id: str, db: Session = Depends(get_db)):
    """Get related documents"""
    relationships = db.query(DocumentRelationship).filter(
        DocumentRelationship.source_document_id == document_id
    ).all()

    return {
        "document_id": document_id,
        "relationships": [
            {
                "target_document_id": str(rel.target_document_id),
                "relationship_type": rel.relationship_type,
                "confidence": rel.confidence
            }
            for rel in relationships
        ]
    }

@app.post("/documents/{document_id}/relationships")
async def create_document_relationship(
    document_id: str,
    target_document_id: str,
    relationship_type: str,
    confidence: float = 0.8,
    db: Session = Depends(get_db)
):
    """Create a relationship between documents"""
    relationship = DocumentRelationship(
        source_document_id=document_id,
        target_document_id=target_document_id,
        relationship_type=relationship_type,
        confidence=confidence
    )

    db.add(relationship)
    db.commit()

    return {"message": "Relationship created successfully"}

@app.get("/statistics")
async def get_statistics(db: Session = Depends(get_db)):
    """Get service statistics"""
    total_documents = db.query(ContentAnalysisJob).count()
    completed_documents = db.query(ContentAnalysisJob).filter(
        ContentAnalysisJob.status == "completed"
    ).count()
    failed_documents = db.query(ContentAnalysisJob).filter(
        ContentAnalysisJob.status == "failed"
    ).count()

    return {
        "total_documents": total_documents,
        "completed_documents": completed_documents,
        "failed_documents": failed_documents,
        "success_rate": completed_documents / total_documents if total_documents > 0 else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
