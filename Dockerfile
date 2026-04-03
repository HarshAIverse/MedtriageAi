# ─────────────────────────────────────────────────────────────────────────────
# Medical Triage Environment — Docker Image
# OpenEnv Submission — HarshAIverse/Meta
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="HarshAIverse"
LABEL description="Medical Triage & Clinical Decision Support — OpenEnv Environment"
LABEL version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        && rm -rf /var/lib/apt/lists/*

# ── Python environment ────────────────────────────────────────────────────────
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY environment.py .
COPY tasks.py .
COPY inference.py .
COPY app.py .
COPY openenv.yaml .

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# ── Runtime configuration ─────────────────────────────────────────────────────
# HF Spaces standard port
EXPOSE 7860

# Environment variable defaults (override at runtime)
ENV PORT=7860 \
    HOST=0.0.0.0 \
    WORKERS=1 \
    LOG_LEVEL=info \
    API_BASE_URL=https://api.openai.com/v1 \
    MODEL_NAME=gpt-4o-mini

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn app:app --host ${HOST} --port ${PORT} --workers ${WORKERS} --log-level ${LOG_LEVEL}"]
