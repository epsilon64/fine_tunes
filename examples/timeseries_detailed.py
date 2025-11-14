"""
Comprehensive Time Series Forecasting Example with Detailed Error Analysis.

This example demonstrates:
1. Loading financial data
2. Preparing features and sequences
3. Training a time series LLM
4. Computing multiple error metrics
5. Analyzing prediction quality
6. Visualizing results
"""

import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Tuple

from src.timeseries.financial_preprocessor import FinancialDataPreprocessor
from src.timeseries.ts_model import AdaptiveTimeSeriesLLM
from src.timeseries.ts_trainer import TimeSeriesTrainer
from src.timeseries.visualization import ForecastVisualizer
from src.core.lora import LoRAConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """Comprehensive forecast evaluation metrics."""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive error metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with error metrics
        """
        # Flatten arrays if needed
        if len(y_true.shape) > 1:
            y_true = y_true.flatten()
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()

        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2

        metrics = {
            # Basic metrics
            'mse': float(np.mean(squared_errors)),
            'rmse': float(np.sqrt(np.mean(squared_errors))),
            'mae': float(np.mean(abs_errors)),
            'mape': float(np.mean(abs_errors / (np.abs(y_true) + 1e-8)) * 100),

            # Median metrics (more robust to outliers)
            'median_ae': float(np.median(abs_errors)),
            'median_ape': float(np.median(abs_errors / (np.abs(y_true) + 1e-8)) * 100),

            # R-squared
            'r2': float(1 - (np.sum(squared_errors) / np.sum((y_true - np.mean(y_true)) ** 2))),

            # Direction accuracy (for trading)
            'direction_accuracy': float(np.mean(np.sign(y_pred) == np.sign(y_true))),

            # Bias
            'mean_error': float(np.mean(errors)),
            'bias': float(np.mean(errors) / (np.mean(np.abs(y_true)) + 1e-8)),

            # Error distribution
            'error_std': float(np.std(errors)),
            'max_error': float(np.max(abs_errors)),

            # Percentiles
            'p50_error': float(np.percentile(abs_errors, 50)),
            'p90_error': float(np.percentile(abs_errors, 90)),
            'p95_error': float(np.percentile(abs_errors, 95)),
            'p99_error': float(np.percentile(abs_errors, 99)),
        }

        return metrics

    @staticmethod
    def calculate_rolling_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        window: int = 50
    ) -> pd.DataFrame:
        """
        Calculate rolling window metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        if len(y_true.shape) > 1:
            y_true = y_true.flatten()
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()

        errors = y_pred - y_true
        abs_errors = np.abs(errors)

        df = pd.DataFrame({
            'true': y_true,
            'pred': y_pred,
            'error': errors,
            'abs_error': abs_errors,
        })

        rolling_metrics = pd.DataFrame({
            'rolling_mae': df['abs_error'].rolling(window).mean(),
            'rolling_mse': (df['error'] ** 2).rolling(window).mean(),
            'rolling_bias': df['error'].rolling(window).mean(),
        })

        return rolling_metrics

    @staticmethod
    def plot_error_analysis(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_dir: str = "./outputs/timeseries_detailed"
    ):
        """
        Create comprehensive error analysis plots.

        Args:
            y_true: True values
            y_pred: Predicted values
            save_dir: Directory to save plots
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if len(y_true.shape) > 1:
            y_true = y_true.flatten()
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()

        errors = y_pred - y_true

        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Forecast Error Analysis', fontsize=16, fontweight='bold')

        # 1. Predictions vs Actuals
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_true.min(), y_true.max()],
                        [y_true.min(), y_true.max()],
                        'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predictions vs Actuals')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Error Distribution
        axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(0, color='r', linestyle='--', lw=2)
        axes[0, 1].axvline(np.mean(errors), color='g', linestyle='--', lw=2,
                          label=f'Mean: {np.mean(errors):.4f}')
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q Plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot (Normality Check)')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Time series comparison (first 200 points)
        n_plot = min(200, len(y_true))
        axes[1, 0].plot(y_true[:n_plot], label='Actual', alpha=0.7, linewidth=2)
        axes[1, 0].plot(y_pred[:n_plot], label='Predicted', alpha=0.7, linewidth=2)
        axes[1, 0].fill_between(range(n_plot),
                                y_true[:n_plot],
                                y_pred[:n_plot],
                                alpha=0.3)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title(f'Time Series Comparison (First {n_plot} Steps)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Rolling MAE
        rolling_mae = pd.Series(np.abs(errors)).rolling(50).mean()
        axes[1, 1].plot(rolling_mae)
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('Rolling Mean Absolute Error (window=50)')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Error over time
        axes[1, 2].scatter(range(len(errors)), errors, alpha=0.3, s=5)
        axes[1, 2].axhline(0, color='r', linestyle='--', lw=2)
        axes[1, 2].axhline(np.mean(errors), color='g', linestyle='--', lw=1)
        axes[1, 2].axhline(np.mean(errors) + 2*np.std(errors), color='orange',
                          linestyle=':', lw=1, label='±2σ')
        axes[1, 2].axhline(np.mean(errors) - 2*np.std(errors), color='orange',
                          linestyle=':', lw=1)
        axes[1, 2].set_xlabel('Time Step')
        axes[1, 2].set_ylabel('Error')
        axes[1, 2].set_title('Errors Over Time')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/error_analysis.png", dpi=300, bbox_inches='tight')
        logger.info(f"Error analysis plot saved to {save_dir}/error_analysis.png")
        plt.close()

    @staticmethod
    def print_metrics_report(metrics: Dict[str, float]):
        """Print formatted metrics report."""
        logger.info("\n" + "=" * 70)
        logger.info("FORECAST EVALUATION METRICS")
        logger.info("=" * 70)

        logger.info("\n📊 Point Forecast Accuracy:")
        logger.info(f"  Mean Squared Error (MSE):          {metrics['mse']:.6f}")
        logger.info(f"  Root Mean Squared Error (RMSE):    {metrics['rmse']:.6f}")
        logger.info(f"  Mean Absolute Error (MAE):         {metrics['mae']:.6f}")
        logger.info(f"  Mean Absolute Percentage Error:    {metrics['mape']:.2f}%")

        logger.info("\n📈 Robust Metrics (Median-based):")
        logger.info(f"  Median Absolute Error:             {metrics['median_ae']:.6f}")
        logger.info(f"  Median Absolute Percentage Error:  {metrics['median_ape']:.2f}%")

        logger.info("\n🎯 Model Quality:")
        logger.info(f"  R-squared (R²):                    {metrics['r2']:.4f}")
        logger.info(f"  Direction Accuracy:                {metrics['direction_accuracy']:.2%}")

        logger.info("\n⚖️  Bias Analysis:")
        logger.info(f"  Mean Error (Bias):                 {metrics['mean_error']:.6f}")
        logger.info(f"  Relative Bias:                     {metrics['bias']:.2%}")
        logger.info(f"  Error Standard Deviation:          {metrics['error_std']:.6f}")

        logger.info("\n🔍 Error Distribution:")
        logger.info(f"  Maximum Error:                     {metrics['max_error']:.6f}")
        logger.info(f"  50th Percentile (Median):          {metrics['p50_error']:.6f}")
        logger.info(f"  90th Percentile:                   {metrics['p90_error']:.6f}")
        logger.info(f"  95th Percentile:                   {metrics['p95_error']:.6f}")
        logger.info(f"  99th Percentile:                   {metrics['p99_error']:.6f}")

        logger.info("\n" + "=" * 70)


def main():
    """Run comprehensive time series forecasting example."""

    logger.info("=" * 70)
    logger.info("COMPREHENSIVE TIME SERIES FORECASTING WITH ERROR ANALYSIS")
    logger.info("=" * 70)

    # Configuration
    TICKER = "AAPL"
    MODEL_NAME = "gpt2"
    SEQUENCE_LENGTH = 30
    PREDICTION_HORIZON = 1
    OUTPUT_DIR = "./outputs/timeseries_detailed"

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Load and prepare data with multiple features
    logger.info(f"\n1. Loading and preparing data for {TICKER}...")
    preprocessor = FinancialDataPreprocessor(
        sequence_length=SEQUENCE_LENGTH,
        prediction_horizon=PREDICTION_HORIZON,
        return_type="log",
    )

    # Load stock data
    stock_data = preprocessor.load_stock_data(
        ticker=TICKER,
        start_date="2020-01-01",
        period="3y",
    )

    # Calculate technical features
    stock_data_with_features = preprocessor.calculate_technical_features(stock_data)

    # Prepare data with multiple features
    data = preprocessor.prepare_data(
        ticker=TICKER,
        start_date="2020-01-01",
        features=["returns", "sma_5", "sma_20", "volatility", "rsi"],
        normalize="standardize",
        train_ratio=0.8,
    )

    logger.info(f"\nData Statistics:")
    logger.info(f"  Total sequences: {len(data['X_train']) + len(data['X_test'])}")
    logger.info(f"  Training sequences: {len(data['X_train'])}")
    logger.info(f"  Test sequences: {len(data['X_test'])}")
    logger.info(f"  Features: {data['X_train'].shape[-1]}")
    logger.info(f"  Sequence length: {SEQUENCE_LENGTH}")
    logger.info(f"  Prediction horizon: {PREDICTION_HORIZON}")

    # 2. Create model
    logger.info("\n2. Creating Adaptive Time Series LLM...")
    model = AdaptiveTimeSeriesLLM(
        model_name=MODEL_NAME,
        d_input=data['X_train'].shape[-1],
        d_output=data['y_train'].shape[-1],
        use_pretrained=True,
        freeze_backbone=False,
        use_temporal_encoding=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    # 3. Apply LoRA (optional but recommended)
    logger.info("\n3. Applying LoRA for efficient training...")
    from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType

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

    # 4. Train model
    logger.info("\n4. Training model...")
    trainer = TimeSeriesTrainer(
        model=model,
        output_dir=OUTPUT_DIR,
    )

    history = trainer.train(
        train_data=data,
        val_data=data,
        epochs=30,
        batch_size=32,
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
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_history.png", dpi=300)
    logger.info(f"Training history plot saved to {OUTPUT_DIR}/training_history.png")
    plt.close()

    # 5. Make predictions
    logger.info("\n5. Generating predictions...")
    predictions = trainer.predict(
        data=data["X_test"],
        batch_size=32,
        steps_ahead=PREDICTION_HORIZON,
    )

    # 6. Calculate comprehensive metrics
    logger.info("\n6. Calculating evaluation metrics...")
    evaluator = ForecastEvaluator()

    y_test = data["y_test"][:, 0, 0]
    y_pred = predictions[:, 0, 0]

    metrics = evaluator.calculate_metrics(y_test, y_pred)

    # Print metrics report
    evaluator.print_metrics_report(metrics)

    # 7. Generate visualizations
    logger.info("\n7. Generating error analysis visualizations...")
    evaluator.plot_error_analysis(y_test, y_pred, OUTPUT_DIR)

    # Use new ForecastVisualizer for comprehensive analysis
    logger.info("\n7b. Generating comprehensive forecast visualizations...")
    visualizer = ForecastVisualizer(output_dir=OUTPUT_DIR)

    # Create comprehensive comparison plot
    visualizer.plot_comprehensive_comparison(
        predictions=y_pred.reshape(-1, 1),
        actuals=y_test.reshape(-1, 1),
        title=f"{TICKER} - Comprehensive Forecast Analysis"
    )

    # Reconstruct prices from returns for price-level visualization
    logger.info("\n7c. Reconstructing and visualizing price forecasts...")
    # Get initial price from the stock data
    initial_price = stock_data['Close'].iloc[len(data['X_train']) * (SEQUENCE_LENGTH + PREDICTION_HORIZON)]

    visualizer.plot_price_reconstruction(
        returns_predictions=y_pred.reshape(-1, 1),
        returns_actuals=y_test.reshape(-1, 1),
        initial_price=initial_price,
        title=f"{TICKER} - Price Forecast vs Realized Prices"
    )

    # 8. Calculate rolling metrics
    logger.info("\n8. Computing rolling window metrics...")
    rolling_metrics = evaluator.calculate_rolling_metrics(y_test, y_pred, window=50)

    # Plot rolling metrics
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Rolling Window Metrics (Window=50)', fontsize=14, fontweight='bold')

    axes[0].plot(rolling_metrics['rolling_mae'], linewidth=2)
    axes[0].set_ylabel('MAE', fontsize=10)
    axes[0].set_title('Rolling Mean Absolute Error')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rolling_metrics['rolling_mse'], linewidth=2, color='orange')
    axes[1].set_ylabel('MSE', fontsize=10)
    axes[1].set_title('Rolling Mean Squared Error')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(rolling_metrics['rolling_bias'], linewidth=2, color='green')
    axes[2].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[2].set_ylabel('Bias', fontsize=10)
    axes[2].set_xlabel('Time Step', fontsize=10)
    axes[2].set_title('Rolling Bias')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rolling_metrics.png", dpi=300)
    logger.info(f"Rolling metrics plot saved to {OUTPUT_DIR}/rolling_metrics.png")
    plt.close()

    # 9. Save metrics to CSV
    logger.info("\n9. Saving metrics to file...")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{OUTPUT_DIR}/forecast_metrics.csv", index=False)
    logger.info(f"Metrics saved to {OUTPUT_DIR}/forecast_metrics.csv")

    # Save predictions
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': y_pred - y_test,
        'abs_error': np.abs(y_pred - y_test),
    })
    results_df.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
    logger.info(f"Predictions saved to {OUTPUT_DIR}/predictions.csv")

    # 10. Trading strategy analysis (if predicting returns)
    logger.info("\n10. Trading Strategy Analysis...")
    logger.info("  Based on direction accuracy:")

    correct_direction = np.sum(np.sign(y_pred) == np.sign(y_test))
    total = len(y_test)
    accuracy = correct_direction / total

    logger.info(f"  Correct direction predictions: {correct_direction}/{total} ({accuracy:.2%})")

    # Simulate simple strategy: go long if predicted positive, short if negative
    strategy_returns = np.where(y_pred > 0, y_test, -y_test)
    buy_and_hold = y_test

    cumulative_strategy = np.cumsum(strategy_returns)
    cumulative_buyhold = np.cumsum(buy_and_hold)

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_strategy, label='Strategy Returns', linewidth=2)
    plt.plot(cumulative_buyhold, label='Buy & Hold', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Returns')
    plt.title(f'{TICKER} - Strategy Performance', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/strategy_performance.png", dpi=300)
    logger.info(f"Strategy performance plot saved to {OUTPUT_DIR}/strategy_performance.png")
    plt.close()

    logger.info(f"\n  Total strategy return: {cumulative_strategy[-1]:.4f}")
    logger.info(f"  Total buy & hold return: {cumulative_buyhold[-1]:.4f}")
    logger.info(f"  Strategy Sharpe ratio: {np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8):.4f}")
    logger.info(f"  Buy & hold Sharpe ratio: {np.mean(buy_and_hold) / (np.std(buy_and_hold) + 1e-8):.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE!")
    logger.info(f"All results saved to: {OUTPUT_DIR}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
