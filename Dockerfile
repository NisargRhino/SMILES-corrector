# Use Python 3.9 base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy your app files into the container
COPY . .

# Upgrade pip and install core dependencies
RUN pip install --upgrade pip

# Install packages from requirements.txt EXCEPT torch/torchtext
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    chembl_structure_pipeline==1.2.2 \
    matplotlib==3.9.4 \
    modin==0.32.0 \
    moses==0.10.0 \
    numpy==1.23.5 \
    pandas==2.2.3 \
    rdkit==2023.3.2 \
    scikit_learn==1.6.1 \
    scipy==1.15.2 \
    seaborn==0.13.2 \
    tueplots==0.2.0

# Install torch and torchtext manually (required for torchtext.legacy)
RUN pip install torch==1.9.0+cpu torchtext==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# Set environment variable for Render port
ENV PORT=10000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]

