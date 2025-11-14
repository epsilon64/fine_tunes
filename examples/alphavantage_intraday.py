"""
Alpha Vantage Intraday Data Training Example.

This example demonstrates using Alpha Vantage API to download true intraday data
(1min, 5min, 15min bars) with extended historical coverage for LLM training.

Requirements:
1. Free Alpha Vantage API key: https://www.alphavantage.co/support/#api-key
2. Set your API key in this script or as environment variable

Data Scale:
- 5min interval, 6 months: ~75,000 bars
- 15min interval, 12 months: ~40,000 bars
- 60min interval, 24 months: ~10,000 bars

This provides true intraday tick-level data (not just daily bars) for better
pattern recognition in high-frequency trading strategies.
"""

import sys
sys.path.append("..")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from src.timeseries.alphavantage_loader import AlphaVantageLoader, print_alphavantage_info
from src.timeseries.financial_preprocessor import FinancialDataPreprocessor
from src.timeseries.ts_model import AdaptiveTimeSeriesLLM
from src.timeseries.ts_trainer import TimeSeriesTrainer
from src.timeseries.visualization import ForecastVisualizer
from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run Alpha Vantage intraday training example."""

    logger.info("=" * 70)
    logger.info("ALPHA VANTAGE INTRADAY DATA TRAINING")
    logger.info("=" * 70)

    # Print API information
    print_alphavantage_info()

    # Configuration
    API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")  # Or set directly: "YOUR_KEY_HERE"

    if not API_KEY:
        logger.error("\n" + "=" * 70)
        logger.error("⚠️  ALPHA VANTAGE API KEY REQUIRED")
        logger.error("=" * 70)
        logger.error("\nPlease set your API key:")
        logger.error("1. Get free key: https://www.alphavantage.co/support/#api-key")
        logger.error("2. Set environment variable:")
        logger.error("   export ALPHAVANTAGE_API_KEY='your_key_here'")
        logger.error("3. Or edit this script and set API_KEY directly")
        logger.error("\nExample with API key:")
        logger.error("   python alphavantage_intraday.py")
        return

    TICKER = "AAPL"
    INTERVAL = "5min"  # Options: 1min, 5min, 15min, 30min, 60min
    MONTHS_BACK = 6    # Download 6 months of history
    OUTPUT_DIR = "./outputs/alphavantage_intraday"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Download intraday data
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: DOWNLOADING INTRADAY DATA FROM ALPHA VANTAGE")
    logger.info("=" * 70)

    logger.info(f"\nConfiguration:")
    logger.info(f"  Ticker: {TICKER}")
    logger.info(f"  Interval: {INTERVAL}")
    logger.info(f"  History: {MONTHS_BACK} months")
    logger.info(f"  Expected bars: ~{MONTHS_BACK * 10000:,} (for {INTERVAL})")

    # Initialize Alpha Vantage loader with caching
    av_loader = AlphaVantageLoader(
        api_key=API_KEY,
        cache_dir="./alphavantage_cache",
        use_cache=True,  # Highly recommended to avoid re-downloads!
    )

    # Download extended history
    intraday_data = av_loader.download_extended_history(
        ticker=TICKER,
        interval=INTERVAL,
        months_back=MONTHS_BACK,
    )

    if intraday_data.empty:
        logger.error("Failed to download data!")
        return

    # Print statistics
    logger.info("\n📊 Data Statistics:")
    logger.info(f"  Total bars: {len(intraday_data):,}")
    logger.info(f"  Date range: {intraday_data.index[0]} to {intraday_data.index[-1]}")
    logger.info(f"  Duration: {(intraday_data.index[-1] - intraday_data.index[0]).days} days")
    logger.info(f"  Average bars/day: {len(intraday_data) / (intraday_data.index[-1] - intraday_data.index[0]).days:.0f}")

    # Visualize raw data
    logger.info("\n2. Visualizing intraday data...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Price chart (last 1000 bars for visibility)
    plot_data = intraday_data.tail(1000)
    axes[0].plot(plot_data.index, plot_data['close'], linewidth=0.5, alpha=0.7)
    axes[0].set_title(f'{TICKER} Intraday Price ({INTERVAL}) - Last 1000 bars', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price ($)')
    axes[0].grid(True, alpha=0.3)

    # Volume chart
    axes[1].bar(plot_data.index, plot_data['volume'], width=0.0001, alpha=0.6)
    axes[1].set_ylabel('Volume')
    axes[1].set_xlabel('Datetime')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/intraday_data_raw.png", dpi=300)
    logger.info(f"  ✓ Raw data plot saved to {OUTPUT_DIR}/intraday_data_raw.png")
    plt.close()

    # 3. Prepare data for training
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: PREPARING DATA FOR TRAINING")
    logger.info("=" * 70)

    # Calculate intraday features
    intraday_data['returns'] = intraday_data['close'].pct_change()
    intraday_data['hl_range'] = (intraday_data['high'] - intraday_data['low']) / intraday_data['close']
    intraday_data['volume_change'] = intraday_data['volume'].pct_change()

    # Moving averages
    intraday_data['sma_20'] = intraday_data['close'].rolling(20).mean()
    intraday_data['sma_50'] = intraday_data['close'].rolling(50).mean()

    # Volatility
    intraday_data['volatility'] = intraday_data['returns'].rolling(20).std()

    # Drop NaN
    intraday_data = intraday_data.dropna()

    logger.info(f"  Cleaned data: {len(intraday_data):,} bars")

    # Create sequences
    SEQUENCE_LENGTH = 100  # 100 bars of history (500 mins = 8.3 hours for 5min)
    PREDICTION_HORIZON = 10  # Predict 10 bars ahead (50 mins for 5min)

    features = ['returns', 'hl_range', 'volume_change', 'sma_20', 'sma_50', 'volatility']

    # Prepare sequences
    X = []
    y = []

    for i in range(len(intraday_data) - SEQUENCE_LENGTH - PREDICTION_HORIZON):
        X.append(intraday_data[features].iloc[i:i+SEQUENCE_LENGTH].values)
        y.append(intraday_data['returns'].iloc[i+SEQUENCE_LENGTH:i+SEQUENCE_LENGTH+PREDICTION_HORIZON].values)

    X = np.array(X)
    y = np.array(y).reshape(-1, PREDICTION_HORIZON, 1)

    # Train/test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    logger.info(f"\n  Training samples: {len(X_train):,}")
    logger.info(f"  Test samples: {len(X_test):,}")
    logger.info(f"  Features: {len(features)}")
    logger.info(f"  Sequence length: {SEQUENCE_LENGTH} bars")
    logger.info(f"  Prediction horizon: {PREDICTION_HORIZON} bars")

    # 4. Create and train model
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: TRAINING MODEL ON INTRADAY DATA")
    logger.info("=" * 70)

    model = AdaptiveTimeSeriesLLM(
        model_name="gpt2",
        d_input=len(features),
        d_output=1,
        use_pretrained=True,
        freeze_backbone=False,
        use_temporal_encoding=True,
    )

    # Apply LoRA for efficient training
    peft_config = PeftLoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    if hasattr(model.backbone, 'transformer'):
        model.backbone.transformer = get_peft_model(model.backbone.transformer, peft_config)

    logger.info(f"  Model: GPT-2 with LoRA")
    logger.info(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train
    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }

    trainer = TimeSeriesTrainer(model=model, output_dir=OUTPUT_DIR)

    history = trainer.train(
        train_data=data_dict,
        val_data=data_dict,
        epochs=20,
        batch_size=32,
        learning_rate=5e-5,
        patience=5,
    )

    # 5. Evaluate
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: EVALUATING INTRADAY PREDICTIONS")
    logger.info("=" * 70)

    predictions = trainer.predict(data=X_test, batch_size=32)

    # Evaluate first horizon
    y_test_flat = y_test[:, 0, 0]
    y_pred_flat = predictions[:, 0, 0]

    mae = np.mean(np.abs(y_pred_flat - y_test_flat))
    rmse = np.sqrt(np.mean((y_pred_flat - y_test_flat) ** 2))
    direction_acc = np.mean(np.sign(y_pred_flat) == np.sign(y_test_flat))

    logger.info(f"\nIntraday Forecast Metrics:")
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  Direction Accuracy: {direction_acc:.2%}")

    # 6. Visualize results
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: CREATING VISUALIZATIONS")
    logger.info("=" * 70)

    visualizer = ForecastVisualizer(output_dir=OUTPUT_DIR)

    # Comprehensive comparison
    visualizer.plot_comprehensive_comparison(
        predictions=y_pred_flat.reshape(-1, 1),
        actuals=y_test_flat.reshape(-1, 1),
        title=f"{TICKER} ({INTERVAL}) - Intraday Forecast Analysis"
    )

    # Multi-horizon analysis if predicting multiple steps
    if PREDICTION_HORIZON > 1:
        visualizer.plot_forecast_horizon_analysis(
            predictions=predictions.squeeze(),
            actuals=y_test.squeeze(),
            title=f"{TICKER} ({INTERVAL}) - Multi-Step Intraday Forecasts"
        )

    # 7. Trading simulation
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: INTRADAY TRADING SIMULATION")
    logger.info("=" * 70)

    # Simple strategy: trade based on predicted direction
    strategy_returns = np.where(y_pred_flat > 0, y_test_flat, -y_test_flat)
    cumulative_strategy = np.cumsum(strategy_returns)
    cumulative_buyhold = np.cumsum(y_test_flat)

    plt.figure(figsize=(14, 6))
    plt.plot(cumulative_strategy, label='Strategy Returns', linewidth=2)
    plt.plot(cumulative_buyhold, label='Buy & Hold', linewidth=2)
    plt.xlabel(f'Bars ({INTERVAL})')
    plt.ylabel('Cumulative Returns')
    plt.title(f'{TICKER} Intraday Strategy Performance ({INTERVAL})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/intraday_strategy.png", dpi=300)
    logger.info(f"  ✓ Strategy plot saved to {OUTPUT_DIR}/intraday_strategy.png")
    plt.close()

    logger.info(f"\n  Strategy return: {cumulative_strategy[-1]:.6f}")
    logger.info(f"  Buy & hold return: {cumulative_buyhold[-1]:.6f}")
    logger.info(f"  Sharpe ratio: {np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8):.4f}")

    # 8. Summary
    logger.info("\n" + "=" * 70)
    logger.info("✓ ALPHA VANTAGE INTRADAY TRAINING COMPLETE!")
    logger.info("=" * 70)

    logger.info(f"\n📊 Summary:")
    logger.info(f"  Data source: Alpha Vantage API")
    logger.info(f"  Interval: {INTERVAL} bars")
    logger.info(f"  Total bars: {len(intraday_data):,}")
    logger.info(f"  Training samples: {len(X_train):,}")
    logger.info(f"  Direction accuracy: {direction_acc:.2%}")
    logger.info(f"\n  Output directory: {OUTPUT_DIR}/")

    logger.info(f"\n💡 Next Steps:")
    logger.info(f"  • Try different intervals (1min for more detail, 60min for more history)")
    logger.info(f"  • Download more months (up to 24 months with slicing)")
    logger.info(f"  • Train on multiple tickers for better generalization")
    logger.info(f"  • Experiment with different sequence lengths")
    logger.info(f"  • Use cached data for faster re-runs (data is already cached!)")


if __name__ == "__main__":
    main()
