# Use a lightweight python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    HOST=0.0.0.0

# Set working directory
WORKDIR /app

# Install system dependencies needed for image processing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY garbage-classification-final-submission-main/ ./garbage-classification-final-submission-main/

# Expose port
EXPOSE 7860

# Set model path env
ENV MODEL_PATH=/app/garbage-classification-final-submission-main/MobileNetV3Large.keras

# Launch application
CMD ["python", "garbage-classification-final-submission-main/app.py"]
