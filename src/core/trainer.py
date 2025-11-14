"""Training utilities and fine-tuner implementation"""

from typing import Optional, Dict, Any, Callable
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FineTuner:
    """
    Fine-tuning orchestrator for language models.

    Supports both full fine-tuning and parameter-efficient methods like LoRA.
    """

    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str = "./outputs",
        use_wandb: bool = False,
    ):
        """
        Initialize fine-tuner.

        Args:
            model: The model to fine-tune
            tokenizer: Tokenizer for the model
            output_dir: Directory to save outputs
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

    def create_training_args(
        self,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        save_steps: int = 500,
        logging_steps: int = 10,
        fp16: bool = True,
        optim: str = "adamw_torch",
        **kwargs
    ) -> TrainingArguments:
        """
        Create training arguments.

        Args:
            num_epochs: Number of training epochs
            batch_size: Per-device batch size
            learning_rate: Learning rate
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Warmup steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
            fp16: Whether to use mixed precision training
            optim: Optimizer type
            **kwargs: Additional arguments for TrainingArguments

        Returns:
            TrainingArguments instance
        """
        args = {
            "output_dir": str(self.output_dir),
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "save_total_limit": 3,
            "fp16": fp16 and torch.cuda.is_available(),
            "optim": optim,
            "report_to": "wandb" if self.use_wandb else "none",
            "load_best_model_at_end": True,
            "gradient_checkpointing": True,
            "ddp_find_unused_parameters": False if isinstance(self.model, PeftModel) else None,
        }

        args.update(kwargs)
        return TrainingArguments(**args)

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_args: Optional[TrainingArguments] = None,
        data_collator: Optional[Callable] = None,
        callbacks: Optional[list] = None,
        **kwargs
    ):
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            training_args: Training arguments (if None, default args are used)
            data_collator: Data collator (if None, default language modeling collator is used)
            callbacks: Training callbacks
            **kwargs: Additional arguments for training_args creation

        Returns:
            Training result
        """
        logger.info("Starting training...")

        # Create training arguments if not provided
        if training_args is None:
            training_args = self.create_training_args(**kwargs)

        # Create data collator if not provided
        if data_collator is None:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal language modeling
            )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Train
        result = trainer.train()

        # Save final model
        self.save_model(str(self.output_dir / "final_model"))

        logger.info("Training completed!")
        logger.info(f"Results: {result}")

        return result

    def save_model(self, path: str):
        """
        Save the model.

        Args:
            path: Path to save the model
        """
        logger.info(f"Saving model to {path}")
        Path(path).mkdir(parents=True, exist_ok=True)

        if isinstance(self.model, PeftModel):
            # Save LoRA adapters
            self.model.save_pretrained(path)
        else:
            # Save full model
            self.model.save_pretrained(path)

        self.tokenizer.save_pretrained(path)
        logger.info("Model saved successfully")

    def load_model(self, path: str):
        """
        Load a saved model.

        Args:
            path: Path to the saved model
        """
        logger.info(f"Loading model from {path}")

        if isinstance(self.model, PeftModel):
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(
                self.model.base_model.model,
                path,
            )
        else:
            # Full model loading would need to recreate model
            logger.warning("Full model loading not implemented. Use ModelLoader instead.")

        logger.info("Model loaded successfully")


class SimpleTrainer:
    """
    Simplified trainer for custom training loops.

    Useful for time series and other specialized training scenarios.
    """

    def __init__(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize simple trainer.

        Args:
            model: Model to train
            optimizer: Optimizer (if None, AdamW is used)
            device: Device to train on
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=2e-4,
            )
        else:
            self.optimizer = optimizer

    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader
            loss_fn: Loss function

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
            else:
                batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)

            # Compute loss
            loss = loss_fn(outputs, batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            dataloader: Evaluation dataloader
            loss_fn: Loss function

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)

                # Forward pass
                outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)

                # Compute loss
                loss = loss_fn(outputs, batch)

                total_loss += loss.item()
                num_batches += 1

        return {
            "eval_loss": total_loss / num_batches,
        }
