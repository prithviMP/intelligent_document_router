FROM postgres:15-alpine

# Set environment variables
ENV POSTGRES_DB=document_db
ENV POSTGRES_USER=admin
ENV POSTGRES_PASSWORD=secret

# Copy initialization script
COPY init.sql /docker-entrypoint-initdb.d/

# Expose port
EXPOSE 5432

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD pg_isready -U admin -d document_db 