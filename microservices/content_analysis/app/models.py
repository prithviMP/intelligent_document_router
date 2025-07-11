from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

class ContentAnalysisJob(Base):
    __tablename__ = 'content_analysis_jobs'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    key_entities = Column(String)
    summary = Column(String)
    sentiment = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow) 