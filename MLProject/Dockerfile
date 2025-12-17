FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (opsional tapi aman untuk sklearn, numpy, pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy seluruh project MLProject
COPY . .

# Command default ketika container dijalankan
CMD ["python", "modelling.py"]
