FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cache friendly)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy core environment files
COPY environment.py tasks.py app.py inference.py openenv.yaml validate.py ./

# Copy server module (required for multi-mode deployment)
COPY server/ ./server/

# Copy static dashboard assets
COPY static/ ./static/

# Expose the Hugging Face Spaces default port
EXPOSE 7860

# Start via server module (satisfies multi-mode deployment spec)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
