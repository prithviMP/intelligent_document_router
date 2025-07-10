"""
API Gateway - Main entry point for the Intelligent Document Classifier and Router
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

# Add libs to path for shared modules
sys.path.append('/app/libs')

from app.routers import documents, auth, health
from app.middleware import logging_middleware, rate_limit_middleware
from app.dependencies import get_current_user
from libs.database.connection import db_manager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting API Gateway...")
    
    # Test database connection
    if not db_manager.test_connection():
        logger.error("Database connection failed")
        raise Exception("Database connection failed")
    
    logger.info("API Gateway started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway...")

# Create FastAPI application
app = FastAPI(
    title="Intelligent Document Classifier and Router",
    description="AI-powered document intelligence that automatically understands, categorizes, and routes business documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add custom middleware
app.middleware("http")(logging_middleware)
app.middleware("http")(rate_limit_middleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(
    documents.router, 
    prefix="/documents", 
    tags=["documents"],
    dependencies=[Depends(get_current_user)]
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Intelligent Document Classifier and Router API",
        "version": "1.0.0",
        "status": "running"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected" if db_manager.test_connection() else "disconnected"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    ) 