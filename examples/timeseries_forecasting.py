"""
Time series forecasting example for financial data.

This example demonstrates using a language model for stock price prediction.
"""

import sys
sys.path.append("..")

from src.timeseries.financial_preprocessor import FinancialDataPreprocessor
from src.timeseries.ts_model import TimeSeriesLLM, AdaptiveTimeSeriesLLM
from src.timeseries.ts_trainer import TimeSeriesTrainer
from src.core.lora import LoRAConfig, apply_lora
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
    """Plot predictions against actual values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual", alpha=0.7)
    plt.plot(y_pred, label="Predicted", alpha=0.7)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    logger.info(f"Plot saved: {title.replace(' ', '_').lower()}.png")


def main():
    """Run time series forecasting example."""

    logger.info("=" * 50)
    logger.info("Time Series Forecasting Example")
    logger.info("=" * 50)

    # Configuration
    TICKER = "AAPL"
    MODEL_NAME = "gpt2"
    SEQUENCE_LENGTH = 30
    PREDICTION_HORIZON = 1
    USE_LORA = True
    OUTPUT_DIR = "./outputs/timeseries"

    # 1. Load and prepare financial data
    logger.info(f"\n1. Loading financial data for {TICKER}...")
    preprocessor = FinancialDataPreprocessor(
        sequence_length=SEQUENCE_LENGTH,
        prediction_horizon=PREDICTION_HORIZON,
        return_type="log",
    )

    data = preprocessor.prepare_data(
        ticker=TICKER,
        start_date="2020-01-01",
        features=["returns"],  # Only use returns
        normalize="standardize",
        train_ratio=0.8,
    )

    logger.info(f"Train sequences: {len(data['X_train'])}")
    logger.info(f"Test sequences: {len(data['X_test'])}")
    logger.info(f"Input shape: {data['X_train'].shape}")
    logger.info(f"Output shape: {data['y_train'].shape}")

    # 2. Create time series model
    logger.info("\n2. Creating time series LLM...")

    # Option A: Basic TimeSeriesLLM
    # model = TimeSeriesLLM(
    #     model_name=MODEL_NAME,
    #     d_input=data['X_train'].shape[-1],  # Number of features
    #     d_output=data['y_train'].shape[-1],  # Prediction dimension
    #     use_pretrained=True,
    #     freeze_backbone=False,
    # )

    # Option B: Adaptive with temporal encoding (recommended)
    model = AdaptiveTimeSeriesLLM(
        model_name=MODEL_NAME,
        d_input=data['X_train'].shape[-1],
        d_output=data['y_train'].shape[-1],
        use_pretrained=True,
        freeze_backbone=False,
        use_temporal_encoding=True,
    )

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # 3. Apply LoRA (optional but recommended)
    if USE_LORA:
        logger.info("\n3. Applying LoRA to model...")
        lora_config = LoRAConfig(
            r=8,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],  # GPT-2 specific
            lora_dropout=0.1,
        )

        # For time series models, we need to apply LoRA to the backbone
        from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType

        peft_config = PeftLoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        # Apply to backbone
        if hasattr(model.backbone, 'transformer'):
            model.backbone.transformer = get_peft_model(model.backbone.transformer, peft_config)
        elif hasattr(model.backbone, 'model'):
            model.backbone.model = get_peft_model(model.backbone.model, peft_config)

        logger.info("LoRA applied successfully")

    # 4. Train model
    logger.info("\n4. Training model...")
    trainer = TimeSeriesTrainer(
        model=model,
        output_dir=OUTPUT_DIR,
    )

    history = trainer.train(
        train_data=data,
        val_data=data,  # Use same dict, it has both train and test
        epochs=20,
        batch_size=32,
        learning_rate=1e-4,
        gradient_clip=1.0,
        save_best=True,
        patience=5,
    )

    logger.info("\n5. Evaluating model...")
    metrics = trainer.evaluate(
        test_data=data,
        batch_size=32,
    )

    # 6. Make predictions
    logger.info("\n6. Making predictions...")
    predictions = trainer.predict(
        data=data["X_test"],
        batch_size=32,
        steps_ahead=PREDICTION_HORIZON,
    )

    # Get actual values for comparison
    y_test = data["y_test"]

    # Flatten for plotting
    y_test_flat = y_test[:, 0, 0] if len(y_test.shape) == 3 else y_test[:, 0]
    y_pred_flat = predictions[:, 0, 0] if len(predictions.shape) == 3 else predictions[:, 0]

    logger.info(f"\nPrediction statistics:")
    logger.info(f"  Mean prediction: {y_pred_flat.mean():.6f}")
    logger.info(f"  Std prediction: {y_pred_flat.std():.6f}")
    logger.info(f"  Mean actual: {y_test_flat.mean():.6f}")
    logger.info(f"  Std actual: {y_test_flat.std():.6f}")

    # Calculate additional metrics
    mse = np.mean((y_pred_flat - y_test_flat) ** 2)
    mae = np.mean(np.abs(y_pred_flat - y_test_flat))
    direction_accuracy = np.mean(np.sign(y_pred_flat) == np.sign(y_test_flat))

    logger.info(f"\nTest Metrics:")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  Direction Accuracy: {direction_accuracy:.2%}")

    # 7. Plot results
    logger.info("\n7. Plotting results...")
    plot_predictions(
        y_test_flat[:200],  # Plot first 200 points
        y_pred_flat[:200],
        title=f"{TICKER} Return Predictions"
    )

    logger.info("\n" + "=" * 50)
    logger.info("Time series forecasting completed!")
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
