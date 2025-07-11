from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

class NotificationLog(Base):
    __tablename__ = 'notification_logs'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recipient = Column(String(100))
    message = Column(String)
    channel = Column(String(50))
    status = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow) 