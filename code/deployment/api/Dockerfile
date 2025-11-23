FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY code/deployment/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY code/deployment/api/main.py .

# Copy code directory with models
COPY code /app/code

# Copy models directory (if exists)
COPY models /app/models

# Create shared data directory
RUN mkdir -p /app/shared_data/uploads /app/shared_data/outputs

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
