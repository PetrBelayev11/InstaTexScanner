


FROM python:3.9-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 libxrender1 libxext6 \
    tesseract-ocr \
    poppler-utils \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt


COPY . /app

EXPOSE 8000
CMD ["uvicorn", "code.deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
