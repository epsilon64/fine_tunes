"""
Test script to verify all examples run without errors.

This script runs each example with minimal settings to catch bugs quickly.
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Change to project root
os.chdir(Path(__file__).parent.parent)


def run_test(script_name: str, test_name: str) -> bool:
    """
    Run a test script and return success/failure.

    Args:
        script_name: Path to script
        test_name: Name for logging

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Testing: {test_name}")
    logger.info(f"{'=' * 70}")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            logger.info(f"✓ {test_name} completed successfully")
            return True
        else:
            logger.error(f"✗ {test_name} failed with return code {result.returncode}")
            logger.error(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"✗ {test_name} timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"✗ {test_name} failed with exception: {e}")
        return False


def main():
    """Run all example tests."""

    logger.info("=" * 70)
    logger.info("RUNNING ALL EXAMPLE TESTS")
    logger.info("=" * 70)

    tests = [
        ("examples/timeseries_forecasting.py", "Time Series Forecasting"),
        ("examples/timeseries_detailed.py", "Time Series Detailed Analysis"),
        ("examples/tick_data_forecasting.py", "Tick Data Forecasting"),
    ]

    results = {}

    for script, name in tests:
        if not Path(script).exists():
            logger.warning(f"⚠ Skipping {name} - script not found: {script}")
            results[name] = None
            continue

        results[name] = run_test(script, name)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for name, result in results.items():
        if result is True:
            logger.info(f"✓ {name}: PASSED")
        elif result is False:
            logger.error(f"✗ {name}: FAILED")
        else:
            logger.warning(f"⚠ {name}: SKIPPED")

    logger.info(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")

    if failed > 0:
        logger.error("\n❌ Some tests failed!")
        sys.exit(1)
    else:
        logger.info("\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
