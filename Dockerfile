FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (for better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Build the vector store at image-build time. Embeddings use FastEmbed
# (ONNX-based, no PyTorch), so this step needs no API key and stays light
# on memory - important for free-tier hosting.
RUN python build_rag.py

EXPOSE 8000

# Render injects a $PORT env var and expects the app to listen on it.
# Falls back to 8000 for local `docker run` without that variable set.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
