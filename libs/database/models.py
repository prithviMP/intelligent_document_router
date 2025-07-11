
"""
SQLAlchemy models for the Intelligent Document Classifier and Router
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class Document(Base):
    """Document model for storing document metadata"""
    __tablename__ = 'documents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_name = Column(String(255), nullable=False)
    storage_path = Column(Text, nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String(100))
    doc_type = Column(String(50))
    confidence = Column(Float, default=0.0)
    status = Column(String(20), default='pending')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True))
    tenant_id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    
    # Relationships
    meta_records = relationship("Metadata", back_populates="document", cascade="all, delete-orphan")
    routing_decisions = relationship("RoutingDecision", back_populates="document", cascade="all, delete-orphan")
    processing_queue = relationship("ProcessingQueue", back_populates="document", cascade="all, delete-orphan")

class Metadata(Base):
    """Metadata model for document analysis results"""
    __tablename__ = 'metadata'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    key_entities = Column(JSONB, default={})
    extracted_text = Column(Text)
    summary = Column(Text)
    topics = Column(ARRAY(String), default=[])
    sentiment_score = Column(Float, default=0.0)
    language = Column(String(10), default='en')
    risk_score = Column(Float, default=0.0)
    compliance_flags = Column(ARRAY(String), default=[])
    related_documents = Column(ARRAY(UUID), default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="meta_records")

class RoutingDecision(Base):
    """Routing decisions for document assignment"""
    __tablename__ = 'routing_decisions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    assigned_to = Column(String(255), nullable=False)
    assigned_team = Column(String(100), nullable=False)
    priority = Column(Integer, default=3)
    confidence = Column(Float, default=0.0)
    rule_id = Column(String(100))
    routing_reason = Column(Text)
    status = Column(String(20), default='pending')
    escalation_deadline = Column(DateTime(timezone=True))
    escalated_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata = Column(JSONB, default={})
    
    # Relationships
    document = relationship("Document", back_populates="routing_decisions")
    
    # Indexes
    __table_args__ = (
        Index('idx_routing_assigned_to', 'assigned_to'),
        Index('idx_routing_team', 'assigned_team'),
        Index('idx_routing_priority', 'priority'),
        Index('idx_routing_status', 'status'),
    )

class ProcessingQueue(Base):
    """Processing queue for document workflow"""
    __tablename__ = 'processing_queue'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    service_name = Column(String(100), nullable=False)
    status = Column(String(20), default='pending')
    priority = Column(Integer, default=3)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    error_message = Column(Text)
    payload = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    scheduled_at = Column(DateTime(timezone=True))
    processed_at = Column(DateTime(timezone=True))
    
    # Relationships
    document = relationship("Document", back_populates="processing_queue")
    
    # Indexes
    __table_args__ = (
        Index('idx_queue_service_status', 'service_name', 'status'),
        Index('idx_queue_priority', 'priority'),
        Index('idx_queue_scheduled', 'scheduled_at'),
    )

class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(255))
    team = Column(String(100))
    role = Column(String(50), default='user')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Indexes
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_team', 'team'),
        Index('idx_user_role', 'role'),
    )

class RoutingRule(Base):
    """Routing rules configuration"""
    __tablename__ = 'routing_rules'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    rule_id = Column(String(100), unique=True, nullable=False)
    conditions = Column(JSONB, nullable=False)
    actions = Column(JSONB, nullable=False)
    priority = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    tenant_id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    
    # Indexes
    __table_args__ = (
        Index('idx_rule_active', 'is_active'),
        Index('idx_rule_priority', 'priority'),
    )

class WorkflowExecution(Base):
    """Workflow execution tracking"""
    __tablename__ = 'workflow_executions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    workflow_name = Column(String(100), nullable=False)
    status = Column(String(20), default='pending')
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    execution_log = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    document = relationship("Document")
    
    # Indexes
    __table_args__ = (
        Index('idx_workflow_status', 'status'),
        Index('idx_workflow_name', 'workflow_name'),
    )
