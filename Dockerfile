FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (for better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Build the vector store at image-build time. Embeddings are a local
# HuggingFace model, so no API key is needed for this step.
RUN python build_rag.py

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
