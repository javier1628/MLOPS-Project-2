# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Force NumPy < 2 for compatibility
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "numpy<2"

# Copy training script
COPY train.py .

# Create directories for outputs
RUN mkdir -p /app/checkpoints /app/logs

# Set default command with best hyperparameters from Project 1 hyperparameter sweep
# Based on systematic search across 5 hyperparameters:
# 1. Learning Rate: 3e-5 (tested: 1e-5, 2e-5, 3e-5, 5e-5, 1e-4)
# 2. Weight Decay: 0.01 (tested: 0.0, 0.01, 0.05, 0.1, 0.2)
# 3. Warmup Steps: 100 (tested: 0, 50, 100, 200, 300)
# 4. Batch Size: 32 (tested: 16, 32, 64, 16×2 accum, 8×4 accum)
# 5. Scheduler: cosine (tested: linear, cosine, cosine+warmup)
CMD ["python", "train.py", \
     "--checkpoint_dir", "/app/checkpoints", \
     "--lr", "3e-5", \
     "--weight_decay", "0.01", \
     "--warmup_steps", "100", \
     "--train_batch_size", "32", \
     "--eval_batch_size", "32", \
     "--max_epochs", "3", \
     "--scheduler_type", "cosine", \
     "--task_name", "mrpc", \
     "--wandb_project", "docker-glue-training", \
     "--wandb_run_name", "docker-best-hyperparams"]

# Labels
LABEL maintainer="Javi <your-email@example.com>"
LABEL description="GLUE training container with best hyperparameters"
LABEL version="1.0"
