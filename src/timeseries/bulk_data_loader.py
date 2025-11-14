"""
Enhanced data loader for pulling maximum historical data.

This module provides functions to download large amounts of historical data
for training LLMs, which require substantial datasets for good performance.

Features:
- Daily data for long-term history (years)
- Bulk downloading with rate limit handling
- Data caching to avoid re-downloads
- Combining multiple tickers efficiently
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging
from pathlib import Path
import time
from datetime import datetime, timedelta
import pickle
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BulkDataLoader:
    """
    Load maximum historical data for multiple tickers.

    Optimized for LLM training which requires large datasets.
    """

    def __init__(
        self,
        cache_dir: str = "./data_cache",
        use_cache: bool = True,
    ):
        """
        Initialize bulk data loader.

        Args:
            cache_dir: Directory to cache downloaded data
            use_cache: Whether to use cached data if available
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

    def _get_cache_path(self, ticker: str, data_type: str = "daily") -> Path:
        """Get cache file path for a ticker."""
        return self.cache_dir / f"{ticker}_{data_type}.pkl"

    def _save_to_cache(self, data: pd.DataFrame, ticker: str, data_type: str = "daily"):
        """Save data to cache."""
        cache_path = self._get_cache_path(ticker, data_type)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"  Cached data for {ticker} to {cache_path}")

    def _load_from_cache(self, ticker: str, data_type: str = "daily") -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_path = self._get_cache_path(ticker, data_type)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"  Loaded {ticker} from cache ({len(data)} records)")
                return data
            except Exception as e:
                logger.warning(f"  Failed to load cache for {ticker}: {e}")
        return None

    def download_daily_history(
        self,
        ticker: str,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Download maximum daily historical data for a ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD) - default is 2010
            end_date: End date (YYYY-MM-DD) - default is today
            force_refresh: Force re-download even if cached

        Returns:
            DataFrame with daily OHLCV data
        """
        # Check cache first
        if self.use_cache and not force_refresh:
            cached_data = self._load_from_cache(ticker, "daily")
            if cached_data is not None:
                return cached_data

        logger.info(f"  Downloading {ticker} from {start_date}...")

        try:
            # Download with yfinance (free, reliable)
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date or datetime.now().strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                logger.warning(f"  No data returned for {ticker}")
                return pd.DataFrame()

            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Standardize column names
            data.columns = data.columns.str.lower()

            # Save to cache
            if self.use_cache:
                self._save_to_cache(data, ticker, "daily")

            years = (data.index[-1] - data.index[0]).days / 365.25
            logger.info(f"  ✓ {ticker}: {len(data)} days ({years:.1f} years)")

            return data

        except Exception as e:
            logger.error(f"  ✗ Failed to download {ticker}: {e}")
            return pd.DataFrame()

    def download_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        delay: float = 0.5,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple tickers with rate limiting.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for download
            end_date: End date for download
            delay: Delay between requests (seconds)

        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        logger.info(f"\nDownloading data for {len(tickers)} tickers...")
        logger.info(f"Period: {start_date} to {end_date or 'today'}")
        logger.info("=" * 70)

        ticker_data = {}
        successful = 0

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"\n[{i}/{len(tickers)}] {ticker}")

            data = self.download_daily_history(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )

            if not data.empty:
                ticker_data[ticker] = data
                successful += 1

            # Rate limiting
            if i < len(tickers):
                time.sleep(delay)

        logger.info("\n" + "=" * 70)
        logger.info(f"✓ Successfully downloaded {successful}/{len(tickers)} tickers")

        # Summary statistics
        if ticker_data:
            total_records = sum(len(df) for df in ticker_data.values())
            avg_records = total_records / len(ticker_data)
            logger.info(f"  Total records: {total_records:,}")
            logger.info(f"  Average per ticker: {avg_records:.0f} records")

        return ticker_data

    def get_combined_dataset(
        self,
        tickers: List[str],
        start_date: str = "2010-01-01",
        calculate_returns: bool = True,
    ) -> pd.DataFrame:
        """
        Download and combine data from multiple tickers into a single dataset.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            calculate_returns: Whether to calculate returns

        Returns:
            Combined DataFrame with all tickers
        """
        ticker_data = self.download_multiple_tickers(
            tickers=tickers,
            start_date=start_date,
        )

        if not ticker_data:
            logger.error("No data downloaded!")
            return pd.DataFrame()

        # Combine all data
        logger.info("\nCombining datasets...")
        combined_rows = []

        for ticker, data in ticker_data.items():
            # Add ticker column
            data_copy = data.copy()
            data_copy['ticker'] = ticker

            # Calculate returns if requested
            if calculate_returns:
                data_copy['returns'] = data_copy['close'].pct_change()
                data_copy['log_returns'] = np.log(data_copy['close'] / data_copy['close'].shift(1))

            combined_rows.append(data_copy)

        combined_df = pd.concat(combined_rows, axis=0)
        combined_df = combined_df.sort_index()

        logger.info(f"✓ Combined dataset: {len(combined_df):,} records")
        logger.info(f"  Tickers: {len(ticker_data)}")
        logger.info(f"  Date range: {combined_df.index[0]} to {combined_df.index[-1]}")

        return combined_df


def get_recommended_tickers(category: str = "tech") -> List[str]:
    """
    Get recommended tickers for training.

    Args:
        category: Category of stocks ('tech', 'finance', 'healthcare', 'diverse', 'sp500_top')

    Returns:
        List of ticker symbols
    """
    tickers_by_category = {
        'tech': [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'CSCO',
            'AVGO', 'QCOM', 'TXN', 'AMAT', 'MU', 'LRCX'
        ],
        'finance': [
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW',
            'AXP', 'USB', 'PNC', 'TFC', 'COF', 'BK'
        ],
        'healthcare': [
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'LLY',
            'DHR', 'BMY', 'AMGN', 'GILD', 'CVS', 'CI'
        ],
        'diverse': [
            # Mix of sectors for maximum diversity
            'AAPL', 'JPM', 'JNJ', 'XOM', 'PG', 'DIS', 'BA', 'HD',
            'MCD', 'KO', 'PEP', 'WMT', 'COST', 'NKE', 'SBUX',
            'V', 'MA', 'PYPL', 'SQ', 'SHOP'
        ],
        'sp500_top': [
            # Top 30 S&P 500 by market cap
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'BRK-B', 'UNH', 'JNJ', 'XOM', 'V', 'PG', 'JPM', 'MA',
            'HD', 'CVX', 'MRK', 'ABBV', 'PEP', 'COST', 'KO', 'AVGO',
            'WMT', 'MCD', 'CSCO', 'DIS', 'ABT', 'TMO', 'VZ'
        ],
    }

    return tickers_by_category.get(category, tickers_by_category['diverse'])


def estimate_training_samples(
    tickers: List[str],
    days_per_ticker: int = 3000,  # ~8-10 years
    sequence_length: int = 30,
    prediction_horizon: int = 5,
) -> Dict[str, int]:
    """
    Estimate how many training samples will be generated.

    Args:
        tickers: List of tickers
        days_per_ticker: Average days of data per ticker
        sequence_length: Length of input sequences
        prediction_horizon: Length of prediction horizon

    Returns:
        Dictionary with estimates
    """
    # Each ticker produces (days - sequence_length - prediction_horizon) samples
    samples_per_ticker = max(0, days_per_ticker - sequence_length - prediction_horizon)
    total_samples = samples_per_ticker * len(tickers)

    # Typically 80/20 split
    train_samples = int(total_samples * 0.8)
    test_samples = total_samples - train_samples

    return {
        'tickers': len(tickers),
        'days_per_ticker': days_per_ticker,
        'samples_per_ticker': samples_per_ticker,
        'total_samples': total_samples,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'data_points': total_samples * sequence_length,  # Actual data points seen
    }


def print_data_statistics(ticker_data: Dict[str, pd.DataFrame]):
    """Print comprehensive statistics about downloaded data."""
    if not ticker_data:
        logger.warning("No data to analyze")
        return

    logger.info("\n" + "=" * 70)
    logger.info("DATA STATISTICS")
    logger.info("=" * 70)

    # Per-ticker stats
    logger.info("\nPer-Ticker Statistics:")
    logger.info(f"{'Ticker':<10} {'Records':<10} {'Start Date':<15} {'End Date':<15} {'Years':<8}")
    logger.info("-" * 70)

    for ticker, data in sorted(ticker_data.items()):
        if not data.empty:
            start = data.index[0].strftime("%Y-%m-%d")
            end = data.index[-1].strftime("%Y-%m-%d")
            years = (data.index[-1] - data.index[0]).days / 365.25
            logger.info(f"{ticker:<10} {len(data):<10} {start:<15} {end:<15} {years:<8.1f}")

    # Aggregate stats
    total_records = sum(len(df) for df in ticker_data.values())
    avg_records = total_records / len(ticker_data)

    # Date range
    all_dates = pd.concat([df.index.to_series() for df in ticker_data.values()])
    overall_start = all_dates.min()
    overall_end = all_dates.max()
    overall_years = (overall_end - overall_start).days / 365.25

    logger.info("\nAggregate Statistics:")
    logger.info(f"  Total tickers: {len(ticker_data)}")
    logger.info(f"  Total records: {total_records:,}")
    logger.info(f"  Average per ticker: {avg_records:.0f}")
    logger.info(f"  Overall date range: {overall_start.strftime('%Y-%m-%d')} to {overall_end.strftime('%Y-%m-%d')}")
    logger.info(f"  Overall span: {overall_years:.1f} years")

    # Estimate training samples
    estimates = estimate_training_samples(
        tickers=list(ticker_data.keys()),
        days_per_ticker=int(avg_records),
    )

    logger.info("\nEstimated Training Samples:")
    logger.info(f"  Samples per ticker: ~{estimates['samples_per_ticker']:,}")
    logger.info(f"  Total samples: ~{estimates['total_samples']:,}")
    logger.info(f"  Training samples: ~{estimates['train_samples']:,}")
    logger.info(f"  Test samples: ~{estimates['test_samples']:,}")
    logger.info(f"  Total data points: ~{estimates['data_points']:,}")

    logger.info("\n" + "=" * 70)


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = BulkDataLoader(cache_dir="./data_cache")

    # Get recommended tickers
    tickers = get_recommended_tickers('sp500_top')[:20]  # Top 20 for demo

    logger.info("Bulk Data Loader Demo")
    logger.info("=" * 70)
    logger.info(f"Downloading {len(tickers)} tickers")
    logger.info(f"Period: 2010-01-01 to today")

    # Download data
    ticker_data = loader.download_multiple_tickers(
        tickers=tickers,
        start_date="2010-01-01",
    )

    # Print statistics
    print_data_statistics(ticker_data)

    # Show how this translates to training data
    logger.info("\nFor context:")
    logger.info("  LLMs typically need 10,000+ samples for good performance")
    logger.info("  GPT-2 was trained on billions of tokens")
    logger.info("  More diverse data (different tickers) improves generalization")
