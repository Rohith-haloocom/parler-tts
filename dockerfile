# Use NVIDIA's CUDA base image with Python support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libasound2 \
    ffmpeg \
    libsndfile1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install essential packages
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install PyTorch with GPU support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn and other dependencies
RUN pip install packaging && \
    pip install flash-attn --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose the port the application runs on
EXPOSE 8400

# Set the FLASK_APP environment variable
ENV FLASK_APP=main.py

# Command to run the Flask app
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=8400"]
