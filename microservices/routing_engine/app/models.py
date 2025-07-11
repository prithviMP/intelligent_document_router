from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

class RoutingDecision(Base):
    __tablename__ = 'routing_decisions'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    assigned_to = Column(String(100))
    priority = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow) 