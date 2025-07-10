"""
Rate limiting middleware for API protection
"""

import time
import logging
from fastapi import Request, HTTPException, status
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

# Simple in-memory rate limiter (use Redis in production)
request_counts = defaultdict(list)
lock = threading.Lock()

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Rate limit configuration
    max_requests = 100  # requests per window
    window_seconds = 60  # time window in seconds
    
    current_time = time.time()
    
    with lock:
        # Clean old requests outside the window
        request_counts[client_ip] = [
            req_time for req_time in request_counts[client_ip]
            if current_time - req_time < window_seconds
        ]
        
        # Check if rate limit exceeded
        if len(request_counts[client_ip]) >= max_requests:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Add current request
        request_counts[client_ip].append(current_time)
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(max_requests)
    response.headers["X-RateLimit-Remaining"] = str(max_requests - len(request_counts[client_ip]))
    response.headers["X-RateLimit-Reset"] = str(int(current_time + window_seconds))
    
    return response 