<<<<<<< HEAD

=======
>>>>>>> 170895b3cacc198e4b5f19af675e4d9b0287efcd
FROM python:3.11-slim

WORKDIR /app

<<<<<<< HEAD
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy shared libraries (will be mounted in docker-compose)
COPY ../../../libs ./libs

# Set environment variables
ENV PYTHONPATH=/app:/app/libs
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/ping || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
=======
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"] 
>>>>>>> 170895b3cacc198e4b5f19af675e4d9b0287efcd
