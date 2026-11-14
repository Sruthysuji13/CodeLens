FROM python:3.11-slim

# Install system dependencies including git (required by GitPython)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and app files
COPY . .

# Ensure storage directories exist
RUN mkdir -p data/repositories data/chroma data/database

# Default port and host for container environments
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# Start server
CMD ["python", "main.py"]
