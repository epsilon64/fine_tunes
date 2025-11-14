"""
Tick Data Forecasting Example.

This example demonstrates:
1. Loading intraday tick data from multiple providers
2. Processing tick-level features
3. Training on high-frequency data
4. Comparing different bar aggregation methods
"""

import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from src.timeseries.tick_data_loader import TickDataLoader, print_provider_comparison
from src.timeseries.financial_preprocessor import FinancialDataPreprocessor
from src.timeseries.ts_model import AdaptiveTimeSeriesLLM
from src.timeseries.ts_trainer import TimeSeriesTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run tick data forecasting example."""

    logger.info("=" * 70)
    logger.info("TICK DATA FORECASTING EXAMPLE")
    logger.info("=" * 70)

    # Print provider comparison
    logger.info("\nAvailable data providers:")
    print_provider_comparison()

    # Configuration
    TICKER = "AAPL"
    PROVIDER = "yahoo"  # Change to 'alphavantage', 'twelvedata', etc. with API key
    API_KEY = None  # Set your API key here if using a paid provider
    INTERVAL = "5min"  # 1min, 5min, 15min, 30min, 60min
    PERIOD = "5d"  # For Yahoo: 1d, 5d, 1mo
    OUTPUT_DIR = "./outputs/tick_forecasting"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Load tick data
    logger.info(f"\n1. Loading {INTERVAL} tick data for {TICKER}...")
    tick_loader = TickDataLoader(provider=PROVIDER, api_key=API_KEY)

    try:
        tick_data = tick_loader.load_intraday_data(
            ticker=TICKER,
            interval=INTERVAL,
            period=PERIOD,
        )

        if tick_data.empty:
            logger.error("No tick data loaded. Please check your configuration.")
            logger.info("\nTips:")
            logger.info("- For Yahoo Finance, try period='5d' with interval='5min'")
            logger.info("- For other providers, set API_KEY and adjust parameters")
            return

        logger.info(f"Loaded {len(tick_data)} ticks")
        logger.info(f"Date range: {tick_data.index[0]} to {tick_data.index[-1]}")

        # Get tick statistics
        tick_stats = tick_loader.get_tick_statistics(tick_data)
        logger.info("\nTick Statistics:")
        logger.info(f"  Total ticks: {tick_stats['total_ticks']}")
        logger.info(f"  Duration: {tick_stats['duration_hours']:.2f} hours")
        logger.info(f"  Mean price: ${tick_stats['mean_price']:.2f}")
        logger.info(f"  Volatility: {tick_stats['volatility']:.4f}")
        logger.info(f"  Total return: {tick_stats['total_return']:.2%}")
        logger.info(f"  Max drawdown: {tick_stats['max_drawdown']:.2%}")

    except Exception as e:
        logger.error(f"Error loading tick data: {e}")
        logger.info("\nTo use this example:")
        logger.info("1. For free: Use provider='yahoo' (no API key needed)")
        logger.info("2. For more data: Get a free API key from Alpha Vantage or Twelve Data")
        logger.info("3. Update API_KEY variable in this script")
        return

    # 2. Visualize raw tick data
    logger.info("\n2. Visualizing tick data...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Price chart
    axes[0].plot(tick_data.index, tick_data['close'], linewidth=1, alpha=0.7)
    axes[0].set_title(f'{TICKER} Tick Data ({INTERVAL} intervals)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price ($)')
    axes[0].grid(True, alpha=0.3)

    # Volume chart
    axes[1].bar(tick_data.index, tick_data['volume'], width=0.0001, alpha=0.6)
    axes[1].set_ylabel('Volume')
    axes[1].set_xlabel('Time')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tick_data_raw.png", dpi=300)
    logger.info(f"Raw tick data plot saved to {OUTPUT_DIR}/tick_data_raw.png")
    plt.close()

    # 3. Compare different bar aggregation methods
    logger.info("\n3. Comparing bar aggregation methods...")
    preprocessor = FinancialDataPreprocessor(
        sequence_length=30,
        prediction_horizon=1,
    )

    # Time bars (already have this from tick_data)
    time_bars = tick_data.copy()
    logger.info(f"  Time bars ({INTERVAL}): {len(time_bars)} bars")

    # Volume bars
    if len(tick_data) > 100:  # Only if we have enough data
        avg_volume = tick_data['volume'].mean()
        volume_bars = preprocessor.aggregate_ticks_to_bars(
            tick_data,
            bar_type="volume",
            bar_size=int(avg_volume * 10),  # Each bar ~10x average volume
        )
        logger.info(f"  Volume bars: {len(volume_bars)} bars")
    else:
        volume_bars = None
        logger.info("  Volume bars: Skipped (insufficient data)")

    # Tick bars (fixed number of ticks per bar)
    if len(tick_data) > 50:
        tick_bars = preprocessor.aggregate_ticks_to_bars(
            tick_data,
            bar_type="tick",
            bar_size=10,  # 10 ticks per bar
        )
        logger.info(f"  Tick bars (10 ticks/bar): {len(tick_bars)} bars")
    else:
        tick_bars = None
        logger.info("  Tick bars: Skipped (insufficient data)")

    # 4. Prepare data with tick-specific features
    logger.info("\n4. Preparing tick data with features...")

    # Use tick data for training
    prepared_data = preprocessor.prepare_tick_data(
        tick_data=time_bars,
        features=[
            'tick_returns',
            'hl_range',
            'volume_intensity',
            'tick_volatility',
            'vwap_deviation',
        ],
        normalize="standardize",
        train_ratio=0.8,
    )

    logger.info(f"  Training samples: {len(prepared_data['X_train'])}")
    logger.info(f"  Test samples: {len(prepared_data['X_test'])}")
    logger.info(f"  Features: {prepared_data['features']}")

    # 5. Train model
    logger.info("\n5. Training time series model on tick data...")

    if len(prepared_data['X_train']) < 20:
        logger.warning("Insufficient training data. Please use a longer period or smaller interval.")
        logger.info("Try: period='5d' with interval='5min' for Yahoo Finance")
        return

    model = AdaptiveTimeSeriesLLM(
        model_name="gpt2",
        d_input=prepared_data['X_train'].shape[-1],
        d_output=prepared_data['y_train'].shape[-1],
        use_pretrained=True,
        freeze_backbone=False,
        use_temporal_encoding=True,
    )

    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = TimeSeriesTrainer(model=model, output_dir=OUTPUT_DIR)

    history = trainer.train(
        train_data=prepared_data,
        val_data=prepared_data,
        epochs=20,
        batch_size=16,
        learning_rate=1e-4,
        patience=5,
    )

    # 6. Evaluate
    logger.info("\n6. Evaluating model...")
    metrics = trainer.evaluate(
        test_data=prepared_data,
        batch_size=16,
    )

    logger.info("\nTest Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.6f}")

    # 7. Make predictions
    logger.info("\n7. Generating predictions...")
    predictions = trainer.predict(
        data=prepared_data["X_test"],
        batch_size=16,
        steps_ahead=1,
    )

    y_test = prepared_data["y_test"][:, 0, 0]
    y_pred = predictions[:, 0, 0]

    # 8. Visualize results
    logger.info("\n8. Creating visualizations...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'Tick Data Forecasting Results - {TICKER} ({INTERVAL})',
                 fontsize=14, fontweight='bold')

    # Training history
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Predictions vs Actuals
    n_plot = min(100, len(y_test))
    axes[1].plot(y_test[:n_plot], label='Actual', linewidth=2, alpha=0.7)
    axes[1].plot(y_pred[:n_plot], label='Predicted', linewidth=2, alpha=0.7)
    axes[1].fill_between(range(n_plot), y_test[:n_plot], y_pred[:n_plot], alpha=0.3)
    axes[1].set_ylabel('Returns')
    axes[1].set_title(f'Predictions vs Actuals (First {n_plot} ticks)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Error distribution
    errors = y_pred - y_test
    axes[2].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[2].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[2].axvline(np.mean(errors), color='g', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(errors):.6f}')
    axes[2].set_xlabel('Prediction Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Error Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tick_forecasting_results.png", dpi=300)
    logger.info(f"Results plot saved to {OUTPUT_DIR}/tick_forecasting_results.png")
    plt.close()

    # 9. Trading simulation
    logger.info("\n9. Simulating trading strategy...")

    # Simple strategy: trade based on predicted direction
    correct_direction = np.sum(np.sign(y_pred) == np.sign(y_test))
    direction_accuracy = correct_direction / len(y_test)

    strategy_returns = np.where(y_pred > 0, y_test, -y_test)
    cumulative_strategy = np.cumsum(strategy_returns)
    cumulative_buyhold = np.cumsum(y_test)

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_strategy, label='Strategy', linewidth=2)
    plt.plot(cumulative_buyhold, label='Buy & Hold', linewidth=2)
    plt.xlabel('Tick Number')
    plt.ylabel('Cumulative Returns')
    plt.title(f'Trading Strategy Performance - {TICKER} ({INTERVAL})',
              fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/trading_strategy.png", dpi=300)
    logger.info(f"Strategy plot saved to {OUTPUT_DIR}/trading_strategy.png")
    plt.close()

    logger.info(f"\n  Direction accuracy: {direction_accuracy:.2%}")
    logger.info(f"  Strategy return: {cumulative_strategy[-1]:.6f}")
    logger.info(f"  Buy & hold return: {cumulative_buyhold[-1]:.6f}")
    logger.info(f"  Sharpe ratio (strategy): {np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8):.4f}")

    # 10. Save results
    logger.info("\n10. Saving results...")
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': errors,
        'strategy_return': strategy_returns,
    })
    results_df.to_csv(f"{OUTPUT_DIR}/tick_predictions.csv", index=False)
    logger.info(f"Predictions saved to {OUTPUT_DIR}/tick_predictions.csv")

    logger.info("\n" + "=" * 70)
    logger.info("TICK DATA FORECASTING COMPLETE!")
    logger.info(f"All results saved to: {OUTPUT_DIR}/")
    logger.info("\nNext steps:")
    logger.info("1. Try different providers (alphavantage, twelvedata) with API keys")
    logger.info("2. Experiment with different intervals (1min, 15min, 1h)")
    logger.info("3. Compare volume bars vs time bars vs tick bars")
    logger.info("4. Add more tick-specific features")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
