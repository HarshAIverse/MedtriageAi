FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy core environment files
COPY environment.py tasks.py app.py inference.py openenv.yaml ./



# Expose the Hugging Face Spaces default port
EXPOSE 7860

# Start the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
