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
    """Metadata model for storing extracted document information"""
    __tablename__ = 'metadata'
    
    id = Column(Integer, primary_key=True)
    doc_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    key_entities = Column(JSONB)  # Extracted names, dates, amounts
    key_phrases = Column(ARRAY(String))
    summary = Column(Text)
    related_docs = Column(ARRAY(UUID(as_uuid=True)))
    risk_score = Column(Float, default=0.0)
    sentiment_score = Column(Float)
    language = Column(String(10))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="meta_records")

class RoutingRule(Base):
    """Routing rules for business logic"""
    __tablename__ = 'routing_rules'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    condition = Column(JSONB, nullable=False)  # IF-THEN logic definition
    assignee = Column(String(100), nullable=False)
    priority = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    tenant_id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    routing_decisions = relationship("RoutingDecision", back_populates="rule")

class RoutingDecision(Base):
    """Routing decisions history"""
    __tablename__ = 'routing_decisions'
    
    id = Column(Integer, primary_key=True)
    doc_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    rule_id = Column(Integer, ForeignKey('routing_rules.id'))
    assigned_to = Column(String(100), nullable=False)
    priority = Column(Integer, default=1)
    decision_reason = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="routing_decisions")
    rule = relationship("RoutingRule", back_populates="routing_decisions")

class User(Base):
    """User model for authentication and assignment"""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    full_name = Column(String(255))
    role = Column(String(50), default='user')
    expertise_tags = Column(ARRAY(String))
    is_active = Column(Boolean, default=True)
    tenant_id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    audit_logs = relationship("AuditLog", back_populates="user")

class Team(Base):
    """Team model for team-based routing"""
    __tablename__ = 'teams'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    members = Column(ARRAY(UUID(as_uuid=True)))
    tenant_id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AuditLog(Base):
    """Audit log for compliance and tracking"""
    __tablename__ = 'audit_log'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(UUID(as_uuid=True))
    details = Column(JSONB)
    ip_address = Column(String(45))  # IPv6 compatible
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")

class ProcessingQueue(Base):
    """Processing queue for tracking document processing"""
    __tablename__ = 'processing_queue'
    
    id = Column(Integer, primary_key=True)
    doc_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    status = Column(String(20), default='queued')
    priority = Column(Integer, default=1)
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    queued_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True))
    
    # Relationships
    document = relationship("Document", back_populates="processing_queue") 