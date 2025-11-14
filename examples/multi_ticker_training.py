"""
Multi-Ticker Time Series Training with Maximum Historical Data.

This example demonstrates training a time series model on multiple stock tickers
with as much historical data as possible (up to 15 years per ticker).

LLMs require substantial training data for good performance:
- Minimum: 10,000+ samples
- Recommended: 50,000+ samples for best results
- This example can generate 100,000+ samples from 20+ tickers

Training on diverse, long-term data helps the model:
- Learn robust patterns that transfer across different stocks
- Capture various market conditions (bull, bear, sideways)
- Generalize better to unseen stocks
- Handle different volatility regimes
"""

import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Dict

from src.timeseries.financial_preprocessor import FinancialDataPreprocessor
from src.timeseries.ts_model import AdaptiveTimeSeriesLLM
from src.timeseries.ts_trainer import TimeSeriesTrainer
from src.timeseries.visualization import ForecastVisualizer
from src.timeseries.bulk_data_loader import (
    BulkDataLoader,
    get_recommended_tickers,
    print_data_statistics,
)
from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTickerDataLoader:
    """Load and combine data from multiple stock tickers."""

    def __init__(
        self,
        tickers: List[str],
        sequence_length: int = 30,
        prediction_horizon: int = 5,
        start_date: str = "2020-01-01",
        features: List[str] = None,
    ):
        """
        Initialize multi-ticker data loader.

        Args:
            tickers: List of stock ticker symbols
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict
            start_date: Start date for data loading
            features: List of feature names to include
        """
        self.tickers = tickers
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.start_date = start_date
        self.features = features or ["returns", "sma_5", "sma_20", "volatility"]

        self.preprocessor = FinancialDataPreprocessor(
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
        )

    def load_ticker_data(self, ticker: str) -> Dict:
        """Load data for a single ticker."""
        try:
            logger.info(f"  Loading {ticker}...")
            data = self.preprocessor.prepare_data(
                ticker=ticker,
                start_date=self.start_date,
                features=self.features,
                normalize="standardize",
                train_ratio=0.8,
            )
            logger.info(f"    ✓ {ticker}: {len(data['X_train'])} train, {len(data['X_test'])} test samples")
            return data
        except Exception as e:
            logger.warning(f"    ✗ Failed to load {ticker}: {e}")
            return None

    def load_all_tickers(self) -> Dict:
        """
        Load and combine data from all tickers.

        Returns:
            Dictionary with combined train/test data
        """
        logger.info(f"\nLoading data for {len(self.tickers)} tickers...")

        all_train_X = []
        all_train_y = []
        all_test_X = []
        all_test_y = []
        successful_tickers = []

        for ticker in self.tickers:
            data = self.load_ticker_data(ticker)

            if data is not None:
                all_train_X.append(data['X_train'])
                all_train_y.append(data['y_train'])
                all_test_X.append(data['X_test'])
                all_test_y.append(data['y_test'])
                successful_tickers.append(ticker)

        if len(successful_tickers) == 0:
            raise ValueError("No tickers loaded successfully!")

        # Concatenate all data
        combined_data = {
            'X_train': np.concatenate(all_train_X, axis=0),
            'y_train': np.concatenate(all_train_y, axis=0),
            'X_test': np.concatenate(all_test_X, axis=0),
            'y_test': np.concatenate(all_test_y, axis=0),
            'features': self.features,
            'scaler': None,  # Each ticker was normalized separately
            'tickers': successful_tickers,
        }

        logger.info(f"\n✓ Combined data from {len(successful_tickers)} tickers:")
        logger.info(f"  Tickers: {', '.join(successful_tickers)}")
        logger.info(f"  Total training samples: {len(combined_data['X_train'])}")
        logger.info(f"  Total test samples: {len(combined_data['X_test'])}")
        logger.info(f"  Features: {len(self.features)}")

        return combined_data

    def load_ticker_separately(self) -> Dict[str, Dict]:
        """
        Load each ticker's data separately for individual evaluation.

        Returns:
            Dictionary mapping ticker -> data dict
        """
        ticker_data = {}

        for ticker in self.tickers:
            data = self.load_ticker_data(ticker)
            if data is not None:
                ticker_data[ticker] = data

        return ticker_data


def evaluate_per_ticker(
    model: AdaptiveTimeSeriesLLM,
    trainer: TimeSeriesTrainer,
    ticker_data: Dict[str, Dict],
    output_dir: str,
) -> pd.DataFrame:
    """
    Evaluate model performance on each ticker separately.

    Args:
        model: Trained model
        trainer: Trainer instance
        ticker_data: Dictionary of ticker -> data
        output_dir: Output directory for visualizations

    Returns:
        DataFrame with metrics per ticker
    """
    logger.info("\n" + "=" * 70)
    logger.info("PER-TICKER EVALUATION")
    logger.info("=" * 70)

    results = []
    visualizer = ForecastVisualizer(output_dir=output_dir)

    for ticker, data in ticker_data.items():
        logger.info(f"\nEvaluating {ticker}...")

        # Make predictions
        predictions = trainer.predict(
            data=data["X_test"],
            batch_size=32,
            steps_ahead=data['y_test'].shape[1],
        )

        y_test = data["y_test"][:, 0, 0]
        y_pred = predictions[:, 0, 0]

        # Calculate metrics
        mae = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        direction_acc = np.mean(np.sign(y_pred) == np.sign(y_test))

        results.append({
            'Ticker': ticker,
            'MAE': mae,
            'RMSE': rmse,
            'Direction_Accuracy': direction_acc,
            'Test_Samples': len(y_test),
        })

        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  Direction Accuracy: {direction_acc:.2%}")

        # Create visualization for this ticker
        try:
            visualizer.plot_comprehensive_comparison(
                predictions=y_pred.reshape(-1, 1),
                actuals=y_test.reshape(-1, 1),
                title=f"{ticker} - Forecast Analysis (Multi-Ticker Model)"
            )
        except Exception as e:
            logger.warning(f"  Visualization failed for {ticker}: {e}")

    results_df = pd.DataFrame(results)
    return results_df


def main():
    """Run multi-ticker training example."""

    logger.info("=" * 70)
    logger.info("MULTI-TICKER TIME SERIES TRAINING")
    logger.info("=" * 70)

    # Configuration
    MODEL_NAME = "gpt2"
    SEQUENCE_LENGTH = 30
    PREDICTION_HORIZON = 5
    OUTPUT_DIR = "./outputs/multi_ticker"
    START_DATE = "2010-01-01"  # Pull 15 years of data!

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Download maximum historical data
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: DOWNLOADING MAXIMUM HISTORICAL DATA")
    logger.info("=" * 70)

    # Get recommended tickers for diverse training
    # Options: 'tech', 'finance', 'healthcare', 'diverse', 'sp500_top'
    TICKERS = get_recommended_tickers('sp500_top')[:30]  # Top 30 S&P 500 stocks

    logger.info(f"\nTargeting {len(TICKERS)} tickers for maximum data diversity")
    logger.info(f"Period: {START_DATE} to today (~15 years)")
    logger.info(f"This will provide 100,000+ training samples for the LLM")

    # Initialize bulk data loader (with caching for faster re-runs)
    bulk_loader = BulkDataLoader(
        cache_dir="./data_cache",
        use_cache=True,  # Cache to avoid re-downloading
    )

    # Download all ticker data
    raw_ticker_data = bulk_loader.download_multiple_tickers(
        tickers=TICKERS,
        start_date=START_DATE,
        delay=0.3,  # Respectful rate limiting
    )

    # Print comprehensive statistics
    print_data_statistics(raw_ticker_data)

    # 2. Process data for training
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: PROCESSING DATA FOR TRAINING")
    logger.info("=" * 70)

    data_loader = MultiTickerDataLoader(
        tickers=list(raw_ticker_data.keys()),  # Use successfully downloaded tickers
        sequence_length=SEQUENCE_LENGTH,
        prediction_horizon=PREDICTION_HORIZON,
        start_date=START_DATE,
        features=["returns", "sma_5", "sma_20", "volatility", "rsi"],
    )

    # Load combined data for training
    combined_data = data_loader.load_all_tickers()

    # Also load ticker data separately for evaluation
    ticker_data = data_loader.load_ticker_separately()

    logger.info("\n✓ Data processing complete!")
    logger.info(f"  Training samples: {len(combined_data['X_train']):,}")
    logger.info(f"  Test samples: {len(combined_data['X_test']):,}")
    logger.info(f"  Total data points: {len(combined_data['X_train']) * SEQUENCE_LENGTH:,}")

    # 3. Create model
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: CREATING MODEL")
    logger.info("=" * 70)

    model = AdaptiveTimeSeriesLLM(
        model_name=MODEL_NAME,
        d_input=combined_data['X_train'].shape[-1],
        d_output=combined_data['y_train'].shape[-1],
        use_pretrained=True,
        freeze_backbone=False,
        use_temporal_encoding=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # 3. Apply LoRA for efficient training
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: APPLYING LoRA")
    logger.info("=" * 70)

    peft_config = PeftLoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # Apply to backbone
    if hasattr(model.backbone, 'transformer'):
        model.backbone.transformer = get_peft_model(model.backbone.transformer, peft_config)
    elif hasattr(model.backbone, 'model'):
        model.backbone.model = get_peft_model(model.backbone.model, peft_config)

    trainable_after_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable after LoRA: {trainable_after_lora:,}")
    logger.info(f"  Reduction: {100 * (1 - trainable_after_lora / trainable_params):.2f}%")

    # 4. Train model on combined data
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: TRAINING ON MULTI-TICKER DATA")
    logger.info("=" * 70)

    trainer = TimeSeriesTrainer(
        model=model,
        output_dir=OUTPUT_DIR,
    )

    history = trainer.train(
        train_data=combined_data,
        val_data=combined_data,
        epochs=30,
        batch_size=64,  # Larger batch since we have more data
        learning_rate=1e-4,
        weight_decay=0.01,
        gradient_clip=1.0,
        save_best=True,
        patience=10,
    )

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Multi-Ticker Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_history.png", dpi=300)
    logger.info(f"Training history saved to {OUTPUT_DIR}/training_history.png")
    plt.close()

    # 5. Evaluate on combined test set
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: EVALUATING ON COMBINED TEST SET")
    logger.info("=" * 70)

    test_metrics = trainer.evaluate(
        test_data=combined_data,
        batch_size=64,
    )

    logger.info("\nCombined Test Metrics:")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.6f}")

    # 6. Evaluate per ticker
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: PER-TICKER EVALUATION")
    logger.info("=" * 70)

    results_df = evaluate_per_ticker(
        model=model,
        trainer=trainer,
        ticker_data=ticker_data,
        output_dir=OUTPUT_DIR,
    )

    # Save results
    results_df.to_csv(f"{OUTPUT_DIR}/per_ticker_results.csv", index=False)
    logger.info(f"\nPer-ticker results saved to {OUTPUT_DIR}/per_ticker_results.csv")

    # 7. Summary visualization
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: CREATING SUMMARY VISUALIZATIONS")
    logger.info("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Multi-Ticker Model Performance', fontsize=14, fontweight='bold')

    # MAE comparison
    axes[0].bar(results_df['Ticker'], results_df['MAE'])
    axes[0].set_title('Mean Absolute Error by Ticker')
    axes[0].set_ylabel('MAE')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)

    # RMSE comparison
    axes[1].bar(results_df['Ticker'], results_df['RMSE'], color='orange')
    axes[1].set_title('RMSE by Ticker')
    axes[1].set_ylabel('RMSE')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

    # Direction Accuracy comparison
    axes[2].bar(results_df['Ticker'], results_df['Direction_Accuracy'] * 100, color='green')
    axes[2].axhline(50, color='r', linestyle='--', linewidth=1, label='Random (50%)')
    axes[2].set_title('Direction Accuracy by Ticker')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/per_ticker_comparison.png", dpi=300)
    logger.info(f"Comparison plot saved to {OUTPUT_DIR}/per_ticker_comparison.png")
    plt.close()

    # 8. Print summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 70)

    logger.info("\nPer-Ticker Results:")
    print(results_df.to_string(index=False))

    logger.info("\n\nAggregate Statistics:")
    logger.info(f"  Average MAE: {results_df['MAE'].mean():.6f}")
    logger.info(f"  Average RMSE: {results_df['RMSE'].mean():.6f}")
    logger.info(f"  Average Direction Accuracy: {results_df['Direction_Accuracy'].mean():.2%}")
    logger.info(f"  Best performing ticker (MAE): {results_df.loc[results_df['MAE'].idxmin(), 'Ticker']}")
    logger.info(f"  Best performing ticker (Direction): {results_df.loc[results_df['Direction_Accuracy'].idxmax(), 'Ticker']}")

    logger.info("\n" + "=" * 70)
    logger.info("MULTI-TICKER TRAINING COMPLETE!")
    logger.info(f"All results saved to: {OUTPUT_DIR}/")
    logger.info("=" * 70)

    logger.info("\nKey Insights:")
    logger.info("✓ Training on multiple tickers improves generalization")
    logger.info("✓ Model learns cross-stock patterns and market dynamics")
    logger.info("✓ Can be applied to new stocks without retraining")
    logger.info("✓ More robust to individual stock anomalies")


if __name__ == "__main__":
    main()
