# Use a slim Python base image
FROM python:3.10-slim

# Set environment variable for TensorFlow optimization and buffering
ENV PYTHONUNBUFFERED=1

# Install necessary system dependencies for image/audio processing
# libglib2.0-0 fixes OpenCV error: libgthread-2.0.so.0 missing
# libgl1 fixes OpenCV error: libGL.so.1 not found
# libsm6, libxrender1, libice6 are for headless OpenCV
# libsndfile1 is CRITICAL for librosa audio handling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libgl1 \
    libwebp-dev \
    libsndfile1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (models, scaler, .npy files, and app.py)
COPY . .

# Cloud Run sets the PORT environment variable automatically, but defining it is good practice
ENV PORT=8080

# Run the application using Gunicorn when the container starts
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 app:app
