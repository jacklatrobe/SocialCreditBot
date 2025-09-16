# Multi-stage build for SocialCreditBot Discord Bot
# Build stage - install dependencies and compile if needed
FROM python:3.11-slim-bookworm AS builder

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Runtime stage - minimal runtime environment
FROM python:3.11-slim-bookworm AS runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Create data directory for SQLite database and logs directory
RUN mkdir -p /app/data /app/logs && chown -R appuser:appuser /app/data /app/logs

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Health check for container orchestration - now simple HTTP call
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health/simple || exit 1

# Expose health check port (FastAPI server port)
EXPOSE 8000

# Use exec form for proper signal handling - FastAPI server via main.py
CMD ["python", "main.py"]