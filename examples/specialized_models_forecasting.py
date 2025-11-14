"""
Time Series Forecasting with Specialized Foundation Models.

This example demonstrates using purpose-built time series models:
- Chronos (Amazon): T5-based time series foundation model
- Lag-Llama: Llama-based probabilistic forecasting model

These models are pre-trained on diverse time series data and often
outperform adapted general LLMs for forecasting tasks.
"""

import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from src.timeseries import (
    FinancialDataPreprocessor,
    TimeSeriesTrainer,
    print_specialized_model_comparison,
    ForecastVisualizer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_chronos_model():
    """Test Chronos model for forecasting."""
    logger.info("\n" + "=" * 70)
    logger.info("TESTING CHRONOS MODEL")
    logger.info("=" * 70)

    try:
        from src.timeseries import ChronosModel

        # Configuration
        TICKER = "AAPL"
        MODEL_SIZE = "tiny"  # tiny, mini, small, base, large
        OUTPUT_DIR = "./outputs/chronos_forecasting"
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        # 1. Prepare data
        logger.info("\n1. Preparing data...")
        preprocessor = FinancialDataPreprocessor(
            sequence_length=30,
            prediction_horizon=5,  # Predict 5 days ahead
        )

        data = preprocessor.prepare_data(
            ticker=TICKER,
            start_date="2022-01-01",
            features=["returns"],
            train_ratio=0.8,
        )

        logger.info(f"  Training samples: {len(data['X_train'])}")
        logger.info(f"  Test samples: {len(data['X_test'])}")

        # 2. Load Chronos model
        logger.info(f"\n2. Loading Chronos-{MODEL_SIZE} model...")
        logger.info("   Note: First run will download the model (~20-200MB depending on size)")

        model = ChronosModel(
            model_size=MODEL_SIZE,
            device="cpu",  # Use "cuda" if you have GPU
        )

        logger.info("   Model loaded successfully!")

        # 3. Generate forecasts (zero-shot, no training needed!)
        logger.info("\n3. Generating zero-shot forecasts...")
        logger.info("   Chronos is pre-trained and works without fine-tuning!")

        predictions = []
        actuals = []

        # Make predictions on test set
        num_samples = min(50, len(data['X_test']))  # Predict on first 50 samples
        for i in range(num_samples):
            # Get context
            context = data['X_test'][i, :, 0]  # (seq_len,)

            # Generate forecast
            forecast = model.predict(
                context=context,
                prediction_length=data['y_test'].shape[1],
                num_samples=10,  # Generate 10 samples for uncertainty
                temperature=1.0,
            )

            # Take median of samples
            pred = forecast.median(dim=0).values.numpy()
            predictions.append(pred)

            # Get actual
            actual = data['y_test'][i, :, 0]
            actuals.append(actual)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # 4. Calculate metrics
        logger.info("\n4. Evaluating forecasts...")

        # For multi-step forecasting, we'll look at each horizon
        for horizon in range(predictions.shape[1]):
            pred_h = predictions[:, horizon]
            actual_h = actuals[:, horizon]

            mae = np.mean(np.abs(pred_h - actual_h))
            mse = np.mean((pred_h - actual_h) ** 2)
            rmse = np.sqrt(mse)

            direction_acc = np.mean(np.sign(pred_h) == np.sign(actual_h))

            logger.info(f"\n  Horizon {horizon + 1}:")
            logger.info(f"    MAE: {mae:.6f}")
            logger.info(f"    RMSE: {rmse:.6f}")
            logger.info(f"    Direction Accuracy: {direction_acc:.2%}")

        # 5. Visualize results
        logger.info("\n5. Creating visualizations...")

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Chronos-{MODEL_SIZE} Forecasts - {TICKER}',
                     fontsize=14, fontweight='bold')

        # Plot first horizon
        axes[0].plot(actuals[:, 0], label='Actual', linewidth=2, alpha=0.7)
        axes[0].plot(predictions[:, 0], label='Predicted', linewidth=2, alpha=0.7)
        axes[0].set_title('1-Day Ahead Forecasts')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Returns')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot errors
        errors = predictions[:, 0] - actuals[:, 0]
        axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[1].set_title('Forecast Errors Distribution')
        axes[1].set_xlabel('Error')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/chronos_forecasts.png", dpi=300)
        logger.info(f"   Plot saved to {OUTPUT_DIR}/chronos_forecasts.png")
        plt.close()

        # Use new ForecastVisualizer for comprehensive analysis
        logger.info("\n5b. Generating comprehensive forecast visualizations...")
        visualizer = ForecastVisualizer(output_dir=OUTPUT_DIR)

        # Create comprehensive comparison plot for first horizon
        visualizer.plot_comprehensive_comparison(
            predictions=predictions[:, 0].reshape(-1, 1),
            actuals=actuals[:, 0].reshape(-1, 1),
            title=f"Chronos-{MODEL_SIZE} - {TICKER} Comprehensive Analysis"
        )

        # Multi-horizon analysis
        if predictions.shape[1] > 1:
            logger.info("\n5c. Creating multi-horizon forecast analysis...")
            horizon_names = [f"H+{i+1}" for i in range(predictions.shape[1])]
            visualizer.plot_forecast_horizon_analysis(
                predictions=predictions,
                actuals=actuals,
                horizon_names=horizon_names,
                title=f"Chronos-{MODEL_SIZE} - Multi-Horizon Analysis"
            )

        # Reconstruct prices from returns
        logger.info("\n5d. Reconstructing and visualizing price forecasts...")
        # Get initial price from the stock data at the start of test period
        test_start_idx = int(len(data['X_train']) + data['X_train'].shape[1])
        stock_data = preprocessor.load_stock_data(ticker=TICKER, start_date="2022-01-01")
        if test_start_idx < len(stock_data):
            initial_price = stock_data['Close'].iloc[test_start_idx]

            visualizer.plot_price_reconstruction(
                returns_predictions=predictions[:, 0].reshape(-1, 1),
                returns_actuals=actuals[:, 0].reshape(-1, 1),
                initial_price=initial_price,
                title=f"Chronos-{MODEL_SIZE} - {TICKER} Price Forecast vs Realized Prices"
            )

        logger.info("\n" + "=" * 70)
        logger.info("✓ Chronos model test completed successfully!")
        logger.info("=" * 70)

        return True

    except ImportError as e:
        logger.error(f"\n✗ Chronos package not installed: {e}")
        logger.info("\nTo install Chronos:")
        logger.info("  pip install git+https://github.com/amazon-science/chronos-forecasting.git")
        logger.info("\nSkipping Chronos test...")
        return False

    except Exception as e:
        logger.error(f"\n✗ Chronos test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lagllama_model():
    """Test Lag-Llama model for forecasting."""
    logger.info("\n" + "=" * 70)
    logger.info("TESTING LAG-LLAMA MODEL")
    logger.info("=" * 70)

    try:
        from src.timeseries import LagLlamaModel

        logger.info("\nNote: Lag-Llama requires:")
        logger.info("  1. GluonTS: pip install gluonts[torch]")
        logger.info("  2. Model checkpoint from: https://github.com/time-series-foundation-models/lag-llama")
        logger.info("\nSkipping Lag-Llama test in this demo...")
        logger.info("See documentation for full setup instructions.")

        return None

    except ImportError as e:
        logger.error(f"\n✗ Lag-Llama dependencies not installed: {e}")
        logger.info("\nTo install Lag-Llama dependencies:")
        logger.info("  pip install gluonts[torch]")
        logger.info("\nSkipping Lag-Llama test...")
        return False

    except Exception as e:
        logger.error(f"\n✗ Lag-Llama test failed: {e}")
        return False


def compare_with_baseline():
    """Compare specialized model with baseline GPT-2 adaptation."""
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: CHRONOS VS GPT-2 ADAPTATION")
    logger.info("=" * 70)

    logger.info("\nKey Differences:")
    logger.info("\n1. Training:")
    logger.info("   - Chronos: Pre-trained on diverse time series → zero-shot ready")
    logger.info("   - GPT-2: Requires fine-tuning on your data")

    logger.info("\n2. Architecture:")
    logger.info("   - Chronos: T5-based, designed for time series")
    logger.info("   - GPT-2: Causal LM, adapted with projection layers")

    logger.info("\n3. Input Format:")
    logger.info("   - Chronos: Direct time series values (tokenized)")
    logger.info("   - GPT-2: Requires embedding layer conversion")

    logger.info("\n4. Performance:")
    logger.info("   - Chronos: Often better out-of-the-box")
    logger.info("   - GPT-2: Can excel with sufficient fine-tuning data")

    logger.info("\n5. Use Cases:")
    logger.info("   - Chronos: Quick prototypes, limited data, zero-shot")
    logger.info("   - GPT-2: Custom tasks, abundant data, transfer learning")


def main():
    """Run specialized models example."""

    logger.info("=" * 70)
    logger.info("SPECIALIZED TIME SERIES FOUNDATION MODELS")
    logger.info("=" * 70)

    # Show model comparison
    print_specialized_model_comparison()

    # Test Chronos
    chronos_result = test_chronos_model()

    # Test Lag-Llama (optional)
    # lagllama_result = test_lagllama_model()

    # Show comparison insights
    if chronos_result:
        compare_with_baseline()

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    if chronos_result:
        logger.info("✓ Chronos model works! Great for:")
        logger.info("  - Quick experiments without training")
        logger.info("  - Limited data scenarios")
        logger.info("  - Baseline comparisons")
        logger.info("  - Production with consistent performance")
    else:
        logger.info("⚠ Chronos not available. Install with:")
        logger.info("  pip install git+https://github.com/amazon-science/chronos-forecasting.git")

    logger.info("\n📚 Next Steps:")
    logger.info("  1. Try different Chronos sizes: tiny → mini → small → base")
    logger.info("  2. Adjust num_samples for uncertainty quantification")
    logger.info("  3. Compare with fine-tuned GPT-2 (timeseries_forecasting.py)")
    logger.info("  4. Use for rapid prototyping and baselines")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
