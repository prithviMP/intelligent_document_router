"""
Health check router for monitoring service status
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from libs.database.connection import get_db
from libs.database.models import User

router = APIRouter()

@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "api_gateway",
        "version": "1.0.0"
    }

@router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Detailed health check with database connectivity"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"disconnected: {str(e)}"
    
    return {
        "status": "healthy",
        "service": "api_gateway",
        "version": "1.0.0",
        "database": db_status,
        "timestamp": "2024-01-01T00:00:00Z"  # TODO: Add proper timestamp
    }

@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    return {
        "status": "ready",
        "service": "api_gateway"
    }

@router.get("/live")
async def liveness_check():
    """Liveness check for Kubernetes"""
    return {
        "status": "alive",
        "service": "api_gateway"
    } 