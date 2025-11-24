FROM python:3.9-slim

WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p /app/converter /app/shared_data/uploads /app/shared_data/outputs

# Copy only necessary Python files
COPY main.py .
COPY converter/latex_converter.py converter/
COPY converter/cnn_detector.py converter/
COPY converter/document_segmenter.py converter/
COPY converter/handwriting_classifier.py converter/
COPY converter/image_detector.py converter/
COPY converter/model_cnn.py converter/
COPY converter/trocr_ocr.py converter/

# Copy model files
COPY converter/models/tiny_cnn_emnist.pth converter/models/

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
