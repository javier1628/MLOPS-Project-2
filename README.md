# GLUE Text Classification Training with Docker

Production-ready training pipeline for GLUE benchmark tasks using PyTorch Lightning, Transformers, and Docker. Train transformer models on text classification tasks with a single command, fully reproducible across platforms.

[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3.11-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)](https://pytorch.org/)

## Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/mlops-glue-training.git
cd mlops-glue-training

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
- [Results](#results)
- [Cloud Deployment](#cloud-deployment)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## Features

- **Dockerized Training**: Fully containerized for reproducibility
- **Single Command Execution**: Start training with one line
- **Highly Configurable**: 20+ hyperparameters via command-line interface
- **Experiment Tracking**: Weights & Biases integration (optional)
- **Reproducible**: Consistent results across different platforms
- **Robust Error Handling**: Automatic detection and handling of common issues
- **Production-Ready**: Includes checkpointing, logging, and monitoring

---

## Prerequisites

- **Docker**: Version 20.10 or later ([installation guide](https://docs.docker.com/get-docker/))
- **Git**: For repository management
- **Hardware**: Minimum 4GB RAM recommended
- **Storage**: Approximately 10GB free disk space

Note: Python installation is not required when using Docker.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mlops-glue-training.git
cd mlops-glue-training

# Build Docker image (first build takes 5-10 minutes)
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

### Available GLUE Tasks

`mrpc`, `sst2`, `cola`, `qnli`, `qqp`, `rte`, `mnli`, `wnli`, `stsb`

### View All Options

```bash
python train.py --help
```

---

## Results

### Expected Performance on MRPC Task

With default hyperparameters:
- **Validation Accuracy**: 85-87%
- **F1 Score**: 89-91%
- **Training Time**: 45-60 minutes on CPU (3 epochs)

### Reproducibility

Results are deterministic when using the same hyperparameters and random seed across different platforms.

---

## Cloud Deployment

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

Note: Batch size reduction to 16 may be necessary due to 4GB RAM limit in standard Codespaces.

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

## Troubleshooting

### Out of Memory Errors

Reduce batch size to accommodate available RAM:

```bash
docker run -it glue-training:latest \
  python train.py --train_batch_size 8
```

For severely constrained environments, use batch size 4 with gradient accumulation:

```bash
docker run -it glue-training:latest \
  python train.py --train_batch_size 4 --accumulate_grad_batches 4
```

### Long Training Duration

CPU training is significantly slower than GPU (approximately 10x). Expected times:
- 1 epoch: 15-20 minutes
- 3 epochs: 45-60 minutes

For quick validation:

```bash
docker run -it glue-training:latest \
  python train.py --max_epochs 1 --no_wandb
```

### Weights & Biases Authentication

The training script includes automatic fallback for W&B authentication failures. To explicitly disable:

```bash
python train.py --no_wandb
```

### Docker Build Failures

Ensure Docker daemon is running:

```bash
docker ps
```

Clear Docker cache if build issues persist:

```bash
docker system prune -a
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

### Training on Different GLUE Tasks

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

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) - Pre-trained models and utilities
- [PyTorch Lightning](https://lightning.ai/) - Training framework
- [GLUE Benchmark](https://gluebenchmark.com/) - Evaluation tasks and datasets
