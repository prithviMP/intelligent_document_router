from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

class ClassificationJob(Base):
    __tablename__ = 'classification_jobs'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    doc_type = Column(String(50))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow) 