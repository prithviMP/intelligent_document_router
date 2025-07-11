"""
Pydantic models for the routing engine service
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
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
    priority: Optional[int] = Field(1, description="Rule priority")
    active: bool = Field(True, description="Whether rule is active")

class ReassignRequest(BaseModel):
    """Request model for document reassignment"""
    document_id: str
    new_assigned_to: str
    new_assigned_team: str
    new_priority: Optional[int] = None
    reason: str

class WorkloadStats(BaseModel):
    """Model for team workload statistics"""
    team: str
    members: Dict[str, int]
    total_active: int
    average_workload: float

class RoutingAnalytics(BaseModel):
    """Model for routing analytics"""
    total_routed: int
    team_distribution: Dict[str, int]
    priority_distribution: Dict[int, int]
    average_confidence: float
    success_rate: float

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

class RoutingMetrics(BaseModel):
    """Model for routing performance metrics"""
    average_routing_time: float
    success_rate: float
    escalation_rate: float
    team_utilization: Dict[str, float]
    bottlenecks: List[str]

class SmartRoutingConfig(BaseModel):
    """Configuration for smart routing features"""
    enable_ml_routing: bool = Field(True, description="Enable ML-based routing")
    workload_balancing: bool = Field(True, description="Enable workload balancing")
    timezone_awareness: bool = Field(True, description="Consider timezone in routing")
    escalation_enabled: bool = Field(True, description="Enable automatic escalation")
    learning_enabled: bool = Field(True, description="Enable learning from feedback")
