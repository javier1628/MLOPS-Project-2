#  Training with Docker


[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3.11-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)](https://pytorch.org/)

## Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MLOPS-Project-2.git
cd MLOPS-Project-2

# Build Docker image
docker build -t glue-training:latest .

# Run training
docker run -it -v $(pwd)/checkpoints:/app/checkpoints glue-training:latest
```

Training completes in approximately 45-60 minutes on CPU. Results are saved to `./checkpoints/`.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

---

## Features

- **Dockerized Training**: Fully containerized for reproducibility
- **Single Command Execution**: Start training with one line
- **Experiment Tracking**: Weights & Biases integration (optional)
- **Robust Error Handling**: Automatic detection and handling of common issues
- **Production-Ready**: Includes checkpointing, logging, and monitoring

---

## Prerequisites

- **Docker**: Version 20.10 or later ([installation guide](https://docs.docker.com/get-docker/))
- **Git**: For repository management

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MLOPS-Project-2.git
cd MLOPS-Project-2

# Build Docker image
docker build -t glue-training:latest .

# Verify successful build
docker images glue-training
```

---

## Usage

### Basic Training with Optimal Hyperparameters

```bash
docker run -it \
  -v $(pwd)/checkpoints:/app/checkpoints \
  glue-training:latest
```

Default configuration:
- Learning Rate: 3e-5
- Weight Decay: 0.01
- Warmup Steps: 100
- Batch Size: 32
- Training Epochs: 3
- Scheduler: Cosine

### Custom Hyperparameters

```bash
docker run -it \
  -v $(pwd)/checkpoints:/app/checkpoints \
  glue-training:latest \
  python train.py \
    --checkpoint_dir /app/checkpoints \
    --lr 5e-5 \
    --weight_decay 0.05 \
    --warmup_steps 200 \
    --train_batch_size 16 \
    --max_epochs 5 \
    --no_wandb
```

### Quick Test Run (Single Epoch)

```bash
docker run -it \
  -v $(pwd)/checkpoints:/app/checkpoints \
  glue-training:latest \
  python train.py \
    --checkpoint_dir /app/checkpoints \
    --max_epochs 1 \
    --no_wandb
```

### With Weights & Biases Tracking

```bash
docker run -it \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -e WANDB_API_KEY=your_api_key_here \
  glue-training:latest \
  python train.py \
    --checkpoint_dir /app/checkpoints \
    --wandb_project "my-glue-project"
```

### Using Docker Compose

```bash
# Default configuration
docker-compose up training

# Without W&B tracking
docker-compose up training-no-wandb

# Quick test
docker-compose --profile test up training-test
```

---

## Configuration

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 2e-5 | Learning rate |
| `--weight_decay` | 0.0 | Weight decay coefficient for AdamW optimizer |
| `--warmup_steps` | 0 | Number of warmup steps for learning rate scheduler |
| `--train_batch_size` | 32 | Training batch size |
| `--max_epochs` | 3 | Maximum number of training epochs |
| `--scheduler_type` | linear | Learning rate scheduler type (linear/cosine) |
| `--checkpoint_dir` | checkpoints | Directory for saving model checkpoints |
| `--task_name` | mrpc | GLUE benchmark task |
| `--no_wandb` | False | Disable Weights & Biases tracking |



### View All Options

```bash
python train.py --help
```

---


### GitHub Codespaces

```bash
# 1. Fork or clone this repository on GitHub
# 2. Create a Codespace (Code → Codespaces → Create codespace on main)
# 3. In the Codespace terminal:

docker build -t glue-training:latest .

docker run -it \
  -v $(pwd)/checkpoints:/app/checkpoints \
  glue-training:latest \
  python train.py \
    --checkpoint_dir /app/checkpoints \
    --train_batch_size 16 \
    --no_wandb
```

### Docker Playground

```bash
# 1. Navigate to labs.play-with-docker.com
# 2. Start a new session and clone the repository

docker build -t glue-training:latest .

# Reduced batch size for 2GB RAM constraint
docker run -it glue-training:latest \
  python train.py \
    --train_batch_size 8 \
    --max_epochs 1 \
    --no_wandb
```

---

## Examples

### Hyperparameter Sweep - Learning Rates

```bash
for lr in 1e-5 2e-5 3e-5 5e-5; do
  docker run -it -v $(pwd)/checkpoints:/app/checkpoints \
    glue-training:latest \
    python train.py --lr $lr --no_wandb
done
```

### Training on Different Tasks

```bash
# Sentiment analysis (SST-2)
docker run -it glue-training:latest \
  python train.py --task_name sst2 --no_wandb

# Question-answer entailment (QNLI)
docker run -it glue-training:latest \
  python train.py --task_name qnli --no_wandb
```

---

## Project Structure

```
mlops-glue-training/
├── train.py              # Main training script
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker orchestration configuration
├── .dockerignore        # Docker build exclusions
├── .gitignore          # Git version control exclusions
├── README.md           # Project documentation
└── checkpoints/        # Model checkpoints (created during training)
```

