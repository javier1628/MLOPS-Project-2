
import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional



def check_and_fix_numpy():
    try:
        import numpy as np
        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
        
        if numpy_version[0] >= 2:
            print(" WARNING: NumPy 2.x detected!")
            print("   PyTorch may have compatibility issues with NumPy 2.x")
            print("   If you encounter errors, run: pip install 'numpy<2'")
            print()
    except Exception as e:
        print(f" Could not check NumPy version: {e}")

check_and_fix_numpy()

warnings.filterwarnings('ignore', message='.*NumPy.*')

import datasets
import evaluate
import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def check_wandb_auth():
    try:
        api = wandb.Api()
        api.viewer()
        return True
    except Exception:
        return False

    
    # Check NumPy
    import numpy as np
    numpy_version = np.__version__
    if numpy_version.startswith('2'):
        print(f" NumPy: {numpy_version} (v2.x detected - may cause issues)")
        print("   Fix if needed: pip install 'numpy<2'")
    else:
        print(f" NumPy: {numpy_version}")
    
    # Check PyTorch
    print(f" PyTorch: {torch.__version__}")
    print(f" Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Check W&B
    if check_wandb_auth():
        print(f" W&B: Authenticated")
    else:
        print(f"  W&B: Not authenticated (will run without W&B)")
        print("   To enable: wandb login")
    
    print("=" * 70)
    print()


# ============================================================================
# DATA MODULE
# ============================================================================

class GLUEDataModule(L.LightningDataModule):
    """Lightning DataModule for GLUE tasks"""
    
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        """Setup datasets for training and validation"""
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        """Download datasets and tokenizers"""
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        """Return validation dataloader(s)"""
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        """Return test dataloader(s)"""
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        """Convert text examples to model input features"""
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding="max_length", truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


# ============================================================================
# MODEL
# ============================================================================

class GLUETransformer(L.LightningModule):    
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        scheduler_type: str = "linear",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})
        return val_loss

    def on_validation_epoch_end(self):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(self.validation_step_outputs):
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            self.validation_step_outputs.clear()
            return loss

        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)

        metrics = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metrics, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        # Choose scheduler type
        if self.hparams.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        else:  # linear
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transformer model on GLUE tasks with W&B tracking"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="distilbert-base-uncased",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="mrpc",
        choices=["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"],
        help="The name of the GLUE task to train on",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        default=2e-5,
        dest="learning_rate",
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Type of learning rate scheduler to use",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization",
    )
    
    # Training configuration
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator type (auto, gpu, cpu, tpu, etc.)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use",
    )
    
    # Checkpoint and logging
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=3,
        help="Save top k model checkpoints",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="glue-training",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    
    return parser.parse_args()


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    # Print startup information
    print_startup_info()
    
    args = parse_args()
    
    # Set random seed for reproducibility
    L.seed_everything(args.seed)
    
    # Generate run name if not provided
    if args.wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_run_name = f"{args.task_name}_{args.model_name_or_path.split('/')[-1]}_{timestamp}"
    
    # Initialize W&B with automatic error handling
    wandb_enabled = False
    if not args.no_wandb:
        try:
            # Try to initialize W&B
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                reinit=True
            )
            wandb_enabled = True
            print(f" W&B run initialized: {args.wandb_run_name}")
        except wandb.errors.CommError as e:
            print("\n  W&B Authentication Error!")
            print("   W&B is not authenticated. Continuing without W&B logging.")
            print("   To enable W&B in future runs:")
            print("   1. Run: wandb login")
            print("   2. Or use --no_wandb flag to skip this warning")
            print(f"   Error details: {str(e)}\n")
            wandb_enabled = False
        except Exception as e:
            print(f"\n  W&B initialization failed: {str(e)}")
            print("   Continuing without W&B logging.\n")
            wandb_enabled = False
    
    # Setup data module
    print(f"\n Loading dataset: {args.task_name}")
    dm = GLUEDataModule(
        model_name_or_path=args.model_name_or_path,
        task_name=args.task_name,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup("fit")
    
    # Create model
    print(f" Initializing model: {args.model_name_or_path}")
    model = GLUETransformer(
        model_name_or_path=args.model_name_or_path,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    
    # Setup checkpoint callback
    checkpoint_dir = Path(args.checkpoint_dir) / args.wandb_run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch{epoch:02d}-val_loss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=args.save_top_k,
        auto_insert_metric_name=False,
    )
    
    # Setup W&B logger
    wandb_logger = None if not wandb_enabled else WandbLogger()
    
    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=True,
    )
    
    # Train
    print(f"\n Starting training for {args.max_epochs} epochs...")
    print(f" Checkpoints will be saved to: {checkpoint_dir}")
    print("=" * 70)
    
    trainer.fit(model, datamodule=dm)
    
    # Get final metrics
    val_metrics = trainer.callback_metrics
    print("\n" + "=" * 70)
    print(" Final Validation Metrics:")
    print("=" * 70)
    for key, value in val_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value.item():.4f}")
    
    # Save final model
    final_model_path = checkpoint_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"\n Final model saved to: {final_model_path}")
    
    # Log best checkpoint info
    if checkpoint_callback.best_model_path:
        print(f" Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"   Best val_loss: {checkpoint_callback.best_model_score:.4f}")
    
    # Finish W&B run
    if wandb_enabled:
        try:
            wandb.finish()
            print(f"\n W&B run finished: {args.wandb_run_name}")
        except Exception as e:
            print(f"\n  Error finishing W&B run: {e}")
    
    return trainer.callback_metrics


if __name__ == "__main__":
    main()
