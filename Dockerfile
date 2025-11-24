# Use lightweight Python
FROM python:3.10-slim

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port (HuggingFace expects 7860 by default, but FastAPI uses 8000; both are fine)
EXPOSE 7860
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
