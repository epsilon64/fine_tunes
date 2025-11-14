"""Training utilities for time series LLMs"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        transform: Optional[callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            X: Input sequences (N, seq_len, features)
            y: Target sequences (N, pred_len, features)
            transform: Optional transform function
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"input": self.X[idx]}

        if self.y is not None:
            item["target"] = self.y[idx]

        if self.transform:
            item = self.transform(item)

        return item


class TimeSeriesTrainer:
    """
    Trainer for time series LLMs.

    Handles training loop, validation, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        output_dir: str = "./ts_outputs",
    ):
        """
        Initialize trainer.

        Args:
            model: Time series model
            device: Device to train on
            output_dir: Directory to save outputs
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

    def train(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        save_best: bool = True,
        patience: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_data: Dictionary with 'X_train' and 'y_train'
            val_data: Dictionary with 'X_test' and 'y_test'
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            gradient_clip: Gradient clipping value
            save_best: Whether to save best model
            patience: Early stopping patience

        Returns:
            Training history
        """
        logger.info("Starting training...")

        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_data["X_train"],
            train_data["y_train"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        if val_data is not None:
            val_dataset = TimeSeriesDataset(
                val_data["X_test"],
                val_data["y_test"],
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            val_loader = None

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=patience // 2,
            verbose=True,
        )

        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader, optimizer, gradient_clip)
            self.history["train_loss"].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.history["val_loss"].append(val_loss)

                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f} - "
                    f"Val Loss: {val_loss:.6f}"
                )

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                    patience_counter = 0
                    logger.info(f"New best model saved (val_loss: {val_loss:.6f})")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}")

        logger.info("Training completed!")
        return self.history

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gradient_clip: float,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            # Move to device
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            predictions = outputs["predictions"]

            # Compute loss (predict next steps)
            # Use the last prediction_horizon steps to match targets
            seq_len = predictions.shape[1]
            target_len = targets.shape[1]

            if seq_len >= target_len:
                # Use last target_len predictions
                pred_for_loss = predictions[:, -target_len:, :]
            else:
                # Pad predictions if needed
                padding = torch.zeros(
                    predictions.shape[0],
                    target_len - seq_len,
                    predictions.shape[2],
                    device=predictions.device
                )
                pred_for_loss = torch.cat([predictions, padding], dim=1)

            loss = self.model.compute_loss(pred_for_loss, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating", leave=False):
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = self.model(inputs)
                predictions = outputs["predictions"]

                # Match prediction length to target length
                seq_len = predictions.shape[1]
                target_len = targets.shape[1]

                if seq_len >= target_len:
                    pred_for_loss = predictions[:, -target_len:, :]
                else:
                    padding = torch.zeros(
                        predictions.shape[0],
                        target_len - seq_len,
                        predictions.shape[2],
                        device=predictions.device
                    )
                    pred_for_loss = torch.cat([predictions, padding], dim=1)

                loss = self.model.compute_loss(pred_for_loss, targets)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def predict(
        self,
        data: np.ndarray,
        batch_size: int = 32,
        steps_ahead: int = 1,
    ) -> np.ndarray:
        """
        Make predictions.

        Args:
            data: Input data (N, seq_len, features)
            batch_size: Batch size
            steps_ahead: Number of steps to predict ahead

        Returns:
            Predictions array
        """
        self.model.eval()

        dataset = TimeSeriesDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting", leave=False):
                inputs = batch["input"].to(self.device)

                # Use autoregressive prediction
                preds = self.model.predict_future(inputs, steps=steps_ahead)

                predictions.append(preds.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        return predictions

    def evaluate(
        self,
        test_data: Dict[str, np.ndarray],
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            test_data: Dictionary with 'X_test' and 'y_test'
            batch_size: Batch size

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model...")

        dataset = TimeSeriesDataset(
            test_data["X_test"],
            test_data["y_test"],
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = self.model(inputs)
                predictions = outputs["predictions"]

                # Match lengths
                seq_len = predictions.shape[1]
                target_len = targets.shape[1]

                if seq_len >= target_len:
                    pred_for_loss = predictions[:, -target_len:, :]
                else:
                    padding = torch.zeros(
                        predictions.shape[0],
                        target_len - seq_len,
                        predictions.shape[2],
                        device=predictions.device
                    )
                    pred_for_loss = torch.cat([predictions, padding], dim=1)

                loss = self.model.compute_loss(pred_for_loss, targets)
                total_loss += loss.item()

                all_predictions.append(pred_for_loss.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Calculate metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)

        # Direction accuracy (for returns prediction)
        pred_direction = np.sign(predictions)
        target_direction = np.sign(targets)
        direction_accuracy = np.mean(pred_direction == target_direction)

        metrics = {
            "test_loss": total_loss / len(dataloader),
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "direction_accuracy": float(direction_accuracy),
        }

        logger.info("Evaluation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")

        return metrics

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.output_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", self.history)
        logger.info(f"Checkpoint loaded from {path}")
