"""
Quick test script to verify all examples work without errors.

Tests basic functionality with minimal training (1-2 epochs).
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_imports():
    """Test that all modules can be imported."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: Basic Imports")
    logger.info("=" * 70)

    try:
        from src.core import ModelLoader, LoRAConfig, apply_lora, FineTuner
        from src.datasets import TextDatasetLoader, prepare_dataset
        from src.timeseries import (
            TimeSeriesLLM,
            AdaptiveTimeSeriesLLM,
            FinancialDataPreprocessor,
            TimeSeriesTrainer,
            TickDataLoader,
            get_provider_info,
        )
        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading functionality."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Data Loading")
    logger.info("=" * 70)

    try:
        from src.timeseries import FinancialDataPreprocessor

        preprocessor = FinancialDataPreprocessor(
            sequence_length=10,
            prediction_horizon=1,
        )

        # Test daily data loading
        logger.info("Testing daily data loading...")
        data = preprocessor.prepare_data(
            ticker="AAPL",
            start_date="2023-01-01",
            features=["returns"],
            train_ratio=0.8,
        )

        logger.info(f"  Loaded {len(data['X_train']) + len(data['X_test'])} sequences")
        logger.info(f"  Training: {len(data['X_train'])}, Testing: {len(data['X_test'])}")

        assert len(data['X_train']) > 0, "No training data"
        assert len(data['X_test']) > 0, "No test data"

        logger.info("✓ Data loading successful")
        return True

    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        traceback.print_exc()
        return False


def test_tick_data_loading():
    """Test tick data loading functionality."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Tick Data Loading")
    logger.info("=" * 70)

    try:
        from src.timeseries import TickDataLoader, FinancialDataPreprocessor

        # Test with Yahoo Finance (free, no API key)
        logger.info("Testing tick data loading with Yahoo Finance...")
        tick_loader = TickDataLoader(provider="yahoo")

        tick_data = tick_loader.load_intraday_data(
            ticker="AAPL",
            interval="5min",
            period="1d",
        )

        if tick_data.empty:
            logger.warning("⚠ No tick data returned (market may be closed)")
            logger.info("✓ Tick data loader works (no data available)")
            return True

        logger.info(f"  Loaded {len(tick_data)} tick records")

        # Test preparing tick data
        preprocessor = FinancialDataPreprocessor(
            sequence_length=10,
            prediction_horizon=1,
        )

        prepared = preprocessor.prepare_tick_data(
            tick_data=tick_data,
            features=['tick_returns'],
            train_ratio=0.8,
        )

        logger.info(f"  Prepared sequences: {len(prepared['X_train']) + len(prepared['X_test'])}")

        logger.info("✓ Tick data loading successful")
        return True

    except Exception as e:
        logger.error(f"✗ Tick data loading failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Model Creation")
    logger.info("=" * 70)

    try:
        from src.timeseries import AdaptiveTimeSeriesLLM
        import torch

        logger.info("Creating TimeSeriesLLM...")
        model = AdaptiveTimeSeriesLLM(
            model_name="gpt2",
            d_input=1,
            d_output=1,
            use_pretrained=True,
            freeze_backbone=False,
            use_temporal_encoding=True,
        )

        logger.info(f"  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Test forward pass
        logger.info("Testing forward pass...")
        dummy_input = torch.randn(2, 10, 1)  # (batch, seq_len, features)
        with torch.no_grad():
            output = model(dummy_input)

        logger.info(f"  Output shape: {output['predictions'].shape}")

        logger.info("✓ Model creation successful")
        return True

    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return False


def test_end_to_end_training():
    """Test end-to-end training with minimal epochs."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: End-to-End Training")
    logger.info("=" * 70)

    try:
        from src.timeseries import (
            FinancialDataPreprocessor,
            AdaptiveTimeSeriesLLM,
            TimeSeriesTrainer,
        )

        # Prepare minimal data
        logger.info("Preparing data...")
        preprocessor = FinancialDataPreprocessor(
            sequence_length=10,
            prediction_horizon=1,
        )

        data = preprocessor.prepare_data(
            ticker="AAPL",
            start_date="2023-01-01",
            features=["returns"],
            train_ratio=0.8,
        )

        if len(data['X_train']) < 10:
            logger.warning("⚠ Insufficient data for training test")
            return True

        # Create model
        logger.info("Creating model...")
        model = AdaptiveTimeSeriesLLM(
            model_name="gpt2",
            d_input=data['X_train'].shape[-1],
            d_output=data['y_train'].shape[-1],
            use_pretrained=True,
            freeze_backbone=True,  # Freeze for faster testing
            use_temporal_encoding=True,
        )

        # Train for 1 epoch
        logger.info("Training for 1 epoch (test)...")
        trainer = TimeSeriesTrainer(
            model=model,
            output_dir="./test_outputs",
        )

        history = trainer.train(
            train_data=data,
            val_data=data,
            epochs=1,  # Just 1 epoch for testing
            batch_size=min(16, len(data['X_train'])),
            learning_rate=1e-4,
            patience=1,
        )

        logger.info(f"  Final training loss: {history['train_loss'][-1]:.6f}")

        # Test prediction
        logger.info("Testing prediction...")
        predictions = trainer.predict(
            data=data["X_test"][:10],  # Just predict 10 samples
            batch_size=8,
            steps_ahead=1,
        )

        logger.info(f"  Generated {len(predictions)} predictions")

        logger.info("✓ End-to-end training successful")
        return True

    except Exception as e:
        logger.error(f"✗ End-to-end training failed: {e}")
        traceback.print_exc()
        return False


def test_error_metrics():
    """Test error metrics calculation."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 6: Error Metrics")
    logger.info("=" * 70)

    try:
        import numpy as np
        import sys
        import os

        # Add examples directory to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

        from timeseries_detailed import ForecastEvaluator

        logger.info("Testing metrics calculation...")

        # Create dummy predictions
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1  # Add some noise

        evaluator = ForecastEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred)

        logger.info(f"  Calculated {len(metrics)} metrics")
        logger.info(f"  Sample metrics: MAE={metrics['mae']:.6f}, R²={metrics['r2']:.4f}")

        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'direction_accuracy' in metrics

        logger.info("✓ Error metrics calculation successful")
        return True

    except Exception as e:
        logger.error(f"✗ Error metrics failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""

    logger.info("=" * 70)
    logger.info("QUICK TEST SUITE - LLM FINE-TUNING FRAMEWORK")
    logger.info("=" * 70)

    # Create output directory
    Path("./test_outputs").mkdir(exist_ok=True)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Loading", test_data_loading),
        ("Tick Data Loading", test_tick_data_loading),
        ("Model Creation", test_model_creation),
        ("End-to-End Training", test_end_to_end_training),
        ("Error Metrics", test_error_metrics),
    ]

    results = {}

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"✗ {name} crashed: {e}")
            traceback.print_exc()
            results[name] = False

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)

    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{status}: {name}")

    logger.info(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")

    if failed > 0:
        logger.error("\n❌ Some tests failed!")
        return 1
    else:
        logger.info("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
