"""
Routing Engine Service - Intelligent Document Routing and Assignment
"""
import os
import sys
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, status, Body, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, create_engine, Column, String, Integer, Float, DateTime, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import asyncio
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./routing_engine.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database Models
class Document(Base):
    __tablename__ = 'documents'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_name = Column(String(255))
    doc_type = Column(String(50))
    confidence = Column(Float, default=0.0)
    status = Column(String(20), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)

class RoutingDecision(Base):
    __tablename__ = 'routing_decisions'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True))
    assigned_to = Column(String(255))
    assigned_team = Column(String(100))
    priority = Column(Integer, default=3)
    confidence = Column(Float, default=0.0)
    rule_id = Column(String(100))
    routing_reason = Column(Text)
    status = Column(String(20), default='pending')
    escalation_deadline = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text)

class User(Base):
    __tablename__ = 'users'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255))
    username = Column(String(100))
    team = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class ProcessingQueue(Base):
    __tablename__ = 'processing_queue'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True))
    service_name = Column(String(100))
    status = Column(String(20), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models
from enum import Enum
class PriorityLevel(int, Enum):
    """Priority levels for document routing"""
    URGENT = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class RoutingStatus(str, Enum):
    """Routing status options"""
    PENDING = "pending"
    ROUTED = "routed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"

class RouteRequest(BaseModel):
    """Request model for document routing"""
    document_id: str = Field(..., description="Unique document identifier")
    doc_type: Optional[str] = Field(None, description="Document type from classification")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Classification confidence")
    urgency_score: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Urgency score")
    custom_rules: Optional[Dict[str, Any]] = Field(None, description="Custom routing rules")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    @validator('document_id')
    def validate_document_id(cls, v):
        try:
            # Try to parse as UUID to validate format
            uuid.UUID(v)
            return v
        except ValueError:
            # If not a valid UUID, return as is (could be any string ID)
            return v

class RouteResponse(BaseModel):
    """Response model for document routing"""
    document_id: str
    assigned_to: str
    assigned_team: str
    priority: int
    confidence: float
    routing_reason: str
    escalation_deadline: datetime
    status: str = "routed"

class RoutingRule(BaseModel):
    """Model for routing rules"""
    id: str
    name: str
    conditions: Dict[str, Any]
    actions: Dict[str, Any]

class RoutingRuleUpdate(BaseModel):
    """Model for updating routing rules"""
    name: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    actions: Optional[Dict[str, Any]] = None

class ReassignRequest(BaseModel):
    """Request model for document reassignment"""
    document_id: str
    new_assigned_to: str
    new_assigned_team: str
    new_priority: Optional[int] = None
    reason: str

class UserCreate(BaseModel):
    """Model for creating users"""
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    username: str = Field(..., min_length=3, max_length=50)
    team: str
    is_active: Optional[bool] = True

class UserUpdate(BaseModel):
    """Model for updating users"""
    email: Optional[str] = Field(None, pattern=r'^[^@]+@[^@]+\.[^@]+$')
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    team: Optional[str] = None
    is_active: Optional[bool] = None

class UserResponse(BaseModel):
    """Response model for users"""
    id: str
    email: str
    username: str
    team: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

class DocumentCreate(BaseModel):
    """Model for creating documents"""
    original_name: str
    doc_type: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    status: Optional[str] = "pending"

class DocumentUpdate(BaseModel):
    """Model for updating documents"""
    original_name: Optional[str] = None
    doc_type: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    status: Optional[str] = None

class DocumentResponse(BaseModel):
    """Response model for documents"""
    id: str
    original_name: str
    doc_type: Optional[str]
    confidence: Optional[float]
    status: str
    created_at: datetime

class RoutingDecisionUpdate(BaseModel):
    """Model for updating routing decisions"""
    assigned_to: Optional[str] = None
    assigned_team: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=5)
    status: Optional[str] = None
    routing_reason: Optional[str] = None

class RoutingMetrics(BaseModel):
    """Model for routing performance metrics"""
    average_routing_time: float
    success_rate: float
    escalation_rate: float
    team_utilization: Dict[str, float]
    bottlenecks: List[str]

class BulkRouteRequest(BaseModel):
    """Request model for bulk document routing"""
    document_ids: List[str] = Field(..., description="List of document IDs to route")
    override_rules: Optional[Dict[str, Any]] = Field(None, description="Override routing rules")
    batch_priority: Optional[int] = Field(None, description="Batch priority override")

class BulkRouteResponse(BaseModel):
    """Response model for bulk document routing"""
    total_documents: int
    successful_routes: int
    failed_routes: int
    routing_results: List[RouteResponse]
    errors: List[str]

class SmartRoutingConfig(BaseModel):
    """Configuration for smart routing features"""
    enable_ml_routing: bool = Field(True, description="Enable ML-based routing")
    workload_balancing: bool = Field(True, description="Enable workload balancing")
    timezone_awareness: bool = Field(True, description="Consider timezone in routing")
    escalation_enabled: bool = Field(True, description="Enable automatic escalation")
    learning_enabled: bool = Field(True, description="Enable learning from feedback")

class EscalationRule(BaseModel):
    """Model for escalation rules"""
    condition: str
    escalation_hours: int
    escalate_to: str
    notification_channels: List[str]

class ContextualRouting(BaseModel):
    """Model for contextual routing based on document content"""
    document_id: str
    entities: Dict[str, List[str]]
    topics: List[str]
    sentiment: float
    urgency_indicators: List[str]
    compliance_requirements: List[str]

app = FastAPI(
    title="Routing Engine Service",
    description="Intelligent document routing and assignment system with comprehensive CRUD operations",
    version="1.0.0"
)

# Configuration
CONTENT_ANALYSIS_URL = os.getenv("CONTENT_ANALYSIS_URL", "http://content_analysis:8003")
CLASSIFICATION_URL = os.getenv("CLASSIFICATION_URL", "http://classification:8001")
WORKFLOW_INTEGRATION_URL = os.getenv("WORKFLOW_INTEGRATION_URL", "http://workflow_integration:8005")

# Default routing rules
DEFAULT_ROUTING_RULES = [
    {
        "id": "contract_legal",
        "name": "Contract Documents to Legal Team",
        "conditions": {
            "doc_type": ["contract", "agreement", "legal_document"],
            "confidence": {"min": 0.8}
        },
        "actions": {
            "assign_to_team": "legal_team",
            "priority": 1,
            "escalation_hours": 24
        }
    },
    {
        "id": "invoice_finance",
        "name": "Invoices to Finance Team",
        "conditions": {
            "doc_type": ["invoice", "bill", "payment_request"],
            "confidence": {"min": 0.7}
        },
        "actions": {
            "assign_to_team": "finance_team",
            "priority": 2,
            "escalation_hours": 48
        }
    },
    {
        "id": "report_management",
        "name": "Reports to Management",
        "conditions": {
            "doc_type": ["report", "analysis", "summary"],
            "confidence": {"min": 0.6}
        },
        "actions": {
            "assign_to_team": "management_team",
            "priority": 3,
            "escalation_hours": 72
        }
    },
    {
        "id": "hr_documents",
        "name": "HR Documents to HR Team",
        "conditions": {
            "doc_type": ["resume", "application", "hr_form"],
            "confidence": {"min": 0.7}
        },
        "actions": {
            "assign_to_team": "hr_team",
            "priority": 2,
            "escalation_hours": 48
        }
    },
    {
        "id": "fallback_general",
        "name": "General Documents",
        "conditions": {
            "doc_type": ["*"],
            "confidence": {"min": 0.0}
        },
        "actions": {
            "assign_to_team": "general_team",
            "priority": 4,
            "escalation_hours": 96
        }
    }
]

# Team workload mapping
TEAM_MEMBERS = {
    "legal_team": ["john.doe@company.com", "jane.smith@company.com"],
    "finance_team": ["alice.johnson@company.com", "bob.wilson@company.com"],
    "management_team": ["ceo@company.com", "cfo@company.com"],
    "hr_team": ["hr.manager@company.com", "hr.assistant@company.com"],
    "general_team": ["support@company.com"]
}

class RoutingEngine:
    """Core routing engine with business logic"""

    def __init__(self):
        self.rules = DEFAULT_ROUTING_RULES

    async def route_document(self, document_data: Dict[str, Any], db: Session) -> RoutingDecision:
        """Route document based on classification and content analysis"""
        try:
            # Apply routing rules
            matched_rule = self._match_rules(document_data)

            # Get team workload
            team_workload = self._get_team_workload(matched_rule["actions"]["assign_to_team"], db)

            # Assign to specific team member
            assigned_user = self._assign_to_user(matched_rule["actions"]["assign_to_team"], team_workload)

            # Convert document_id to UUID if it's a string
            doc_id = document_data["document_id"]
            if isinstance(doc_id, str):
                try:
                    doc_uuid = uuid.UUID(doc_id)
                except ValueError:
                    # If not a valid UUID, generate a new one
                    doc_uuid = uuid.uuid4()
                    logger.warning(f"Invalid UUID format for document_id: {doc_id}, generated new UUID: {doc_uuid}")
            else:
                doc_uuid = doc_id

            # Create routing decision
            routing_decision = RoutingDecision(
                document_id=doc_uuid,
                assigned_to=assigned_user,
                assigned_team=matched_rule["actions"]["assign_to_team"],
                priority=matched_rule["actions"]["priority"],
                rule_id=matched_rule["id"],
                confidence=document_data.get("confidence", 0.0),
                escalation_deadline=datetime.utcnow() + timedelta(hours=matched_rule["actions"]["escalation_hours"]),
                routing_reason=f"Matched rule: {matched_rule['name']}",
                metadata_json=str({
                    "doc_type": document_data.get("doc_type"),
                    "extracted_entities": document_data.get("extracted_entities", {}),
                    "risk_score": document_data.get("risk_score", 0.0)
                })
            )

            db.add(routing_decision)
            db.commit()
            db.refresh(routing_decision)

            logger.info(f"Document {doc_uuid} routed to {assigned_user}")

            return routing_decision

        except Exception as e:
            logger.error(f"Error routing document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")

    def _match_rules(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Match document against routing rules"""
        doc_type = document_data.get("doc_type", "unknown")
        confidence = document_data.get("confidence", 0.0)

        for rule in self.rules:
            conditions = rule["conditions"]

            # Check doc_type condition
            if doc_type in conditions["doc_type"] or "*" in conditions["doc_type"]:
                # Check confidence condition
                if confidence >= conditions["confidence"]["min"]:
                    logger.info(f"Document matched rule: {rule['name']}")
                    return rule

        # Return fallback rule if no match
        return self.rules[-1]  # Last rule should be fallback

    def _get_team_workload(self, team: str, db: Session) -> Dict[str, int]:
        """Get current workload for team members"""
        team_members = TEAM_MEMBERS.get(team, [])
        workload = {}

        for member in team_members:
            # Count active assignments
            active_count = db.query(RoutingDecision).filter(
                and_(
                    RoutingDecision.assigned_to == member,
                    RoutingDecision.status.in_(["pending", "in_progress"])
                )
            ).count()

            workload[member] = active_count

        return workload

    def _assign_to_user(self, team: str, workload: Dict[str, int]) -> str:
        """Assign document to user with lowest workload"""
        if not workload:
            return TEAM_MEMBERS.get(team, ["support@company.com"])[0]

        # Find user with minimum workload
        min_workload_user = min(workload.items(), key=lambda x: x[1])
        return min_workload_user[0]

# Initialize routing engine
routing_engine = RoutingEngine()

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    """Create database tables on startup"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Routing Engine Service",
        "version": "1.0.0",
        "status": "running",
        "description": "Intelligent document routing and assignment system",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "route_document": "/route",
            "routing_decisions": "/routes",
            "rules": "/rules",
            "users": "/users",
            "documents": "/documents",
            "analytics": "/analytics/routing"
        },
        "message": "Service is running successfully!"
    }

# Health check endpoint
@app.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"message": "pong from Routing Engine", "status": "healthy"}

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Detailed health check"""
    try:
        db.execute("SELECT 1")
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# ===== ROUTING ENDPOINTS =====

@app.post("/route", response_model=RouteResponse)
async def route_document(
    request: RouteRequest = Body(...),
    db: Session = Depends(get_db)
):
    """Route a document to appropriate team/user"""
    try:
        # Get document classification if not provided
        if not request.doc_type or not request.confidence:
            try:
                async with httpx.AsyncClient() as client:
                    classification_response = await client.get(
                        f"{CLASSIFICATION_URL}/document/{request.document_id}/classification"
                    )
                    if classification_response.status_code == 200:
                        classification_data = classification_response.json()
                        request.doc_type = classification_data.get("doc_type", request.doc_type)
                        request.confidence = classification_data.get("confidence", request.confidence or 0.0)
            except Exception as e:
                logger.warning(f"Classification service unavailable: {str(e)}")

        # Get content analysis
        content_analysis = {}
        try:
            async with httpx.AsyncClient() as client:
                content_response = await client.get(
                    f"{CONTENT_ANALYSIS_URL}/analyze/{request.document_id}"
                )
                if content_response.status_code == 200:
                    content_analysis = content_response.json()
        except Exception as e:
            logger.warning(f"Content analysis failed: {str(e)}")

        # Prepare document data for routing
        document_data = {
            "document_id": request.document_id,
            "doc_type": request.doc_type or "unknown",
            "confidence": request.confidence or 0.0,
            "extracted_entities": content_analysis.get("entities", {}),
            "risk_score": content_analysis.get("risk_score", 0.0),
            "urgency_score": request.urgency_score
        }

        # Route the document
        routing_decision = await routing_engine.route_document(document_data, db)

        # Send notification
        try:
            await send_notification(routing_decision)
        except Exception as e:
            logger.warning(f"Notification failed: {str(e)}")

        return RouteResponse(
            document_id=request.document_id,
            assigned_to=routing_decision.assigned_to,
            assigned_team=routing_decision.assigned_team,
            priority=routing_decision.priority,
            confidence=routing_decision.confidence,
            routing_reason=routing_decision.routing_reason,
            escalation_deadline=routing_decision.escalation_deadline,
            status="routed"
        )

    except Exception as e:
        logger.error(f"Routing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/route/{document_id}", response_model=RouteResponse)
async def get_routing_decision(document_id: str = Path(...), db: Session = Depends(get_db)):
    """Get routing decision for a document"""
    try:
        # Try to parse as UUID first
        try:
            doc_uuid = uuid.UUID(document_id)
            routing_decision = db.query(RoutingDecision).filter(
                RoutingDecision.document_id == doc_uuid
            ).first()
        except ValueError:
            # If not a valid UUID, search by string
            routing_decision = db.query(RoutingDecision).filter(
                RoutingDecision.document_id == document_id
            ).first()

        if not routing_decision:
            raise HTTPException(status_code=404, detail="Routing decision not found")

        return RouteResponse(
            document_id=str(routing_decision.document_id),
            assigned_to=routing_decision.assigned_to,
            assigned_team=routing_decision.assigned_team,
            priority=routing_decision.priority,
            confidence=routing_decision.confidence,
            routing_reason=routing_decision.routing_reason,
            escalation_deadline=routing_decision.escalation_deadline,
            status=routing_decision.status
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting routing decision: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/routes", response_model=List[RouteResponse])
async def list_routing_decisions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    team: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    priority: Optional[int] = Query(None, ge=1, le=5),
    db: Session = Depends(get_db)
):
    """List all routing decisions with optional filtering"""
    try:
        query = db.query(RoutingDecision)

        if team:
            query = query.filter(RoutingDecision.assigned_team == team)
        if status:
            query = query.filter(RoutingDecision.status == status)
        if priority:
            query = query.filter(RoutingDecision.priority == priority)

        decisions = query.offset(skip).limit(limit).all()

        return [
            RouteResponse(
                document_id=str(decision.document_id),
                assigned_to=decision.assigned_to,
                assigned_team=decision.assigned_team,
                priority=decision.priority,
                confidence=decision.confidence,
                routing_reason=decision.routing_reason,
                escalation_deadline=decision.escalation_deadline,
                status=decision.status
            )
            for decision in decisions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/route/{document_id}", response_model=RouteResponse)
async def update_routing_decision(
    document_id: str = Path(...),
    update_data: RoutingDecisionUpdate = Body(...),
    db: Session = Depends(get_db)
):
    """Update a routing decision"""
    try:
        # Try to parse as UUID first
        try:
            doc_uuid = uuid.UUID(document_id)
            routing_decision = db.query(RoutingDecision).filter(
                RoutingDecision.document_id == doc_uuid
            ).first()
        except ValueError:
            routing_decision = db.query(RoutingDecision).filter(
                RoutingDecision.document_id == document_id
            ).first()

        if not routing_decision:
            raise HTTPException(status_code=404, detail="Routing decision not found")

        # Update fields
        for field, value in update_data.dict(exclude_unset=True).items():
            setattr(routing_decision, field, value)

        routing_decision.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(routing_decision)

        return RouteResponse(
            document_id=str(routing_decision.document_id),
            assigned_to=routing_decision.assigned_to,
            assigned_team=routing_decision.assigned_team,
            priority=routing_decision.priority,
            confidence=routing_decision.confidence,
            routing_reason=routing_decision.routing_reason,
            escalation_deadline=routing_decision.escalation_deadline,
            status=routing_decision.status
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/route/{document_id}")
async def delete_routing_decision(
    document_id: str = Path(...),
    db: Session = Depends(get_db)
):
    """Delete a routing decision"""
    try:
        # Try to parse as UUID first
        try:
            doc_uuid = uuid.UUID(document_id)
            routing_decision = db.query(RoutingDecision).filter(
                RoutingDecision.document_id == doc_uuid
            ).first()
        except ValueError:
            routing_decision = db.query(RoutingDecision).filter(
                RoutingDecision.document_id == document_id
            ).first()

        if not routing_decision:
            raise HTTPException(status_code=404, detail="Routing decision not found")

        db.delete(routing_decision)
        db.commit()

        return {"message": "Routing decision deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reassign", response_model=Dict[str, Any])
async def reassign_document(
    request: ReassignRequest = Body(...),
    db: Session = Depends(get_db)
):
    """Reassign document to different user/team"""
    try:
        # Try to parse as UUID first
        try:
            doc_uuid = uuid.UUID(request.document_id)
            routing_decision = db.query(RoutingDecision).filter(
                RoutingDecision.document_id == doc_uuid
            ).first()
        except ValueError:
            routing_decision = db.query(RoutingDecision).filter(
                RoutingDecision.document_id == request.document_id
            ).first()

        if not routing_decision:
            raise HTTPException(status_code=404, detail="Routing decision not found")

        # Update routing decision
        routing_decision.assigned_to = request.new_assigned_to
        routing_decision.assigned_team = request.new_assigned_team
        routing_decision.priority = request.new_priority or routing_decision.priority
        routing_decision.routing_reason = f"Reassigned: {request.reason}"
        routing_decision.updated_at = datetime.utcnow()

        db.commit()

        # Send notification
        try:
            await send_notification(routing_decision)
        except Exception as e:
            logger.warning(f"Notification failed: {str(e)}")

        return {"message": "Document reassigned successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# ===== RULES ENDPOINTS =====

@app.post("/rules", response_model=Dict[str, Any])
async def add_routing_rule(rule: RoutingRule = Body(...)):
    """Add a new routing rule"""
    try:
        new_rule = {
            "id": rule.id,
            "name": rule.name,
            "conditions": rule.conditions,
            "actions": rule.actions
        }

        routing_engine.rules.insert(-1, new_rule)  # Insert before fallback rule

        return {"message": "Rule added successfully", "rule": new_rule}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rules", response_model=List[Dict[str, Any]])
async def get_routing_rules():
    """Get all routing rules"""
    return routing_engine.rules

@app.get("/rules/{rule_id}", response_model=Dict[str, Any])
async def get_routing_rule(rule_id: str = Path(...)):
    """Get a specific routing rule"""
    for rule in routing_engine.rules:
        if rule["id"] == rule_id:
            return rule
    raise HTTPException(status_code=404, detail="Rule not found")

@app.put("/rules/{rule_id}", response_model=Dict[str, Any])
async def update_routing_rule(
    rule_id: str = Path(...),
    rule_update: RoutingRuleUpdate = Body(...)
):
    """Update a routing rule"""
    try:
        for i, rule in enumerate(routing_engine.rules):
            if rule["id"] == rule_id:
                # Update fields
                update_data = rule_update.dict(exclude_unset=True)
                for field, value in update_data.items():
                    rule[field] = value

                routing_engine.rules[i] = rule
                return {"message": "Rule updated successfully", "rule": rule}

        raise HTTPException(status_code=404, detail="Rule not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/rules/{rule_id}")
async def delete_routing_rule(rule_id: str = Path(...)):
    """Delete a routing rule"""
    try:
        for i, rule in enumerate(routing_engine.rules):
            if rule["id"] == rule_id:
                # Don't allow deletion of fallback rule
                if rule_id == "fallback_general":
                    raise HTTPException(status_code=400, detail="Cannot delete fallback rule")

                del routing_engine.rules[i]
                return {"message": "Rule deleted successfully"}

        raise HTTPException(status_code=404, detail="Rule not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== USER MANAGEMENT ENDPOINTS =====

@app.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate = Body(...), db: Session = Depends(get_db)):
    """Create a new user"""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.email == user.email) | (User.username == user.username)
        ).first()

        if existing_user:
            raise HTTPException(status_code=400, detail="User with this email or username already exists")

        new_user = User(
            email=user.email,
            username=user.username,
            team=user.team,
            is_active=user.is_active
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return UserResponse(
            id=str(new_user.id),
            email=new_user.email,
            username=new_user.username,
            team=new_user.team,
            is_active=new_user.is_active,
            created_at=new_user.created_at,
            updated_at=new_user.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    team: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    db: Session = Depends(get_db)
):
    """List all users with optional filtering"""
    try:
        query = db.query(User)

        if team:
            query = query.filter(User.team == team)
        if is_active is not None:
            query = query.filter(User.is_active == is_active)

        users = query.offset(skip).limit(limit).all()

        return [
            UserResponse(
                id=str(user.id),
                email=user.email,
                username=user.username,
                team=user.team,
                is_active=user.is_active,
                created_at=user.created_at,
                updated_at=user.updated_at
            )
            for user in users
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str = Path(...), db: Session = Depends(get_db)):
    """Get a specific user"""
    try:
        user_uuid = uuid.UUID(user_id)
        user = db.query(User).filter(User.id == user_uuid).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return UserResponse(
            id=str(user.id),
            email=user.email,
            username=user.username,
            team=user.team,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str = Path(...),
    user_update: UserUpdate = Body(...),
    db: Session = Depends(get_db)
):
    """Update a user"""
    try:
        user_uuid = uuid.UUID(user_id)
        user = db.query(User).filter(User.id == user_uuid).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update fields
        for field, value in user_update.dict(exclude_unset=True).items():
            setattr(user, field, value)

        user.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(user)

        return UserResponse(
            id=str(user.id),
            email=user.email,
            username=user.username,
            team=user.team,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/users/{user_id}", response_model=UserResponse)
async def patch_user(
    user_id: str = Path(...),
    user_update: UserUpdate = Body(...),
    db: Session = Depends(get_db)
):
    """Partially update a user (same as PUT in this case)"""
    return await update_user(user_id, user_update, db)

@app.delete("/users/{user_id}")
async def delete_user(user_id: str = Path(...), db: Session = Depends(get_db)):
    """Delete a user"""
    try:
        user_uuid = uuid.UUID(user_id)
        user = db.query(User).filter(User.id == user_uuid).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        db.delete(user)
        db.commit()

        return {"message": "User deleted successfully"}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# ===== DOCUMENT ENDPOINTS =====

@app.post("/documents", response_model=DocumentResponse)
async def create_document(document: DocumentCreate = Body(...), db: Session = Depends(get_db)):
    """Create a new document"""
    try:
        new_document = Document(
            original_name=document.original_name,
            doc_type=document.doc_type,
            confidence=document.confidence,
            status=document.status
        )

        db.add(new_document)
        db.commit()
        db.refresh(new_document)

        return DocumentResponse(
            id=str(new_document.id),
            original_name=new_document.original_name,
            doc_type=new_document.doc_type,
            confidence=new_document.confidence,
            status=new_document.status,
            created_at=new_document.created_at
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    doc_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """List all documents with optional filtering"""
    try:
        query = db.query(Document)

        if doc_type:
            query = query.filter(Document.doc_type == doc_type)
        if status:
            query = query.filter(Document.status == status)

        documents = query.offset(skip).limit(limit).all()

        return [
            DocumentResponse(
                id=str(doc.id),
                original_name=doc.original_name,
                doc_type=doc.doc_type,
                confidence=doc.confidence,
                status=doc.status,
                created_at=doc.created_at
            )
            for doc in documents
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str = Path(...), db: Session = Depends(get_db)):
    """Get a specific document"""
    try:
        doc_uuid = uuid.UUID(document_id)
        document = db.query(Document).filter(Document.id == doc_uuid).first()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return DocumentResponse(
            id=str(document.id),
            original_name=document.original_name,
            doc_type=document.doc_type,
            confidence=document.confidence,
            status=document.status,
            created_at=document.created_at
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str = Path(...),
    document_update: DocumentUpdate = Body(...),
    db: Session = Depends(get_db)
):
    """Update a document"""
    try:
        doc_uuid = uuid.UUID(document_id)
        document = db.query(Document).filter(Document.id == doc_uuid).first()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Update fields
        for field, value in document_update.dict(exclude_unset=True).items():
            setattr(document, field, value)

        db.commit()
        db.refresh(document)

        return DocumentResponse(
            id=str(document.id),
            original_name=document.original_name,
            doc_type=document.doc_type,
            confidence=document.confidence,
            status=document.status,
            created_at=document.created_at
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str = Path(...), db: Session = Depends(get_db)):
    """Delete a document"""
    try:
        doc_uuid = uuid.UUID(document_id)
        document = db.query(Document).filter(Document.id == doc_uuid).first()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        db.delete(document)
        db.commit()

        return {"message": "Document deleted successfully"}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# ===== ANALYTICS AND WORKLOAD ENDPOINTS =====

@app.get("/workload/{team}", response_model=Dict[str, Any])
async def get_team_workload(team: str = Path(...), db: Session = Depends(get_db)):
    """Get workload statistics for a team"""
    try:
        workload = routing_engine._get_team_workload(team, db)

        total_active = sum(workload.values())
        avg_workload = total_active / len(workload) if workload else 0

        return {
            "team": team,
            "members": workload,
            "total_active": total_active,
            "average_workload": avg_workload
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/routing", response_model=Dict[str, Any])
async def get_routing_analytics(db: Session = Depends(get_db)):
    """Get routing analytics and statistics"""
    try:
        # Get routing statistics
        total_routed = db.query(RoutingDecision).count()

        # Get routing by team
        team_stats = db.query(
            RoutingDecision.assigned_team,
            func.count(RoutingDecision.id).label('count')
        ).group_by(RoutingDecision.assigned_team).all()

        # Get routing by priority
        priority_stats = db.query(
            RoutingDecision.priority,
            func.count(RoutingDecision.id).label('count')
        ).group_by(RoutingDecision.priority).all()

        # Get average confidence
        avg_confidence = db.query(func.avg(RoutingDecision.confidence)).scalar()

        return {
            "total_routed": total_routed,
            "team_distribution": {team: count for team, count in team_stats},
            "priority_distribution": {priority: count for priority, count in priority_stats},
            "average_confidence": avg_confidence or 0.0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def send_notification(routing_decision: RoutingDecision):
    """Send notification about routing decision"""
    try:
        notification_data = {
            "type": "document_routed",
            "document_id": str(routing_decision.document_id),
            "assigned_to": routing_decision.assigned_to,
            "assigned_team": routing_decision.assigned_team,
            "priority": routing_decision.priority,
            "message": f"Document routed to {routing_decision.assigned_to}",
            "escalation_deadline": routing_decision.escalation_deadline.isoformat()
        }

        async with httpx.AsyncClient() as client:
            await client.post(
                f"{WORKFLOW_INTEGRATION_URL}/notify",
                json=notification_data
            )

    except Exception as e:
        logger.error(f"Notification failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5001,
        reload=True
    )