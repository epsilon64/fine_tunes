"""
Enhanced Alpha Vantage intraday data loader.

Alpha Vantage provides true intraday data (1min, 5min, 15min, 30min, 60min bars)
with up to 2 years of historical data using extended API calls.

Free Tier Limits:
- 5 API calls per minute
- 500 API calls per day
- Full intraday history requires multiple calls (slices)

Premium Tier:
- Higher rate limits
- More concurrent requests
- Real-time data updates
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import requests
import time
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaVantageLoader:
    """
    Enhanced Alpha Vantage loader for maximum intraday data.

    Features:
    - Automatic slicing for extended history (up to 2 years)
    - Rate limit handling with automatic delays
    - Data caching to conserve API calls
    - Progress tracking for long downloads
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: str = "./alphavantage_cache",
        use_cache: bool = True,
    ):
        """
        Initialize Alpha Vantage loader.

        Args:
            api_key: Your Alpha Vantage API key (get free at https://www.alphavantage.co/support/#api-key)
            cache_dir: Directory to cache downloaded data
            use_cache: Whether to use cached data if available
        """
        if not api_key:
            raise ValueError(
                "Alpha Vantage API key required!\n"
                "Get free key at: https://www.alphavantage.co/support/#api-key"
            )

        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.base_url = "https://www.alphavantage.co/query"

        # Rate limiting (5 calls/min for free tier)
        self.calls_per_minute = 5
        self.call_times = []

    def _get_cache_path(self, cache_key: str, interval: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}_{interval}_intraday.pkl"

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str, interval: str):
        """Save data to cache."""
        cache_path = self._get_cache_path(cache_key, interval)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"  ✓ Cached {cache_key} {interval} data ({len(data)} bars)")

    def _load_from_cache(self, cache_key: str, interval: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_path = self._get_cache_path(cache_key, interval)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"  ✓ Loaded {cache_key} from cache ({len(data)} bars)")
                return data
            except Exception as e:
                logger.warning(f"  Failed to load cache: {e}")
        return None

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()

        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if current_time - t < 60]

        # If we've hit the limit, wait
        if len(self.call_times) >= self.calls_per_minute:
            wait_time = 60 - (current_time - self.call_times[0]) + 1
            if wait_time > 0:
                logger.info(f"  Rate limit: waiting {wait_time:.0f}s...")
                time.sleep(wait_time)
                self.call_times = []

        # Record this call
        self.call_times.append(current_time)

    def download_intraday(
        self,
        ticker: str,
        interval: str = "5min",
        outputsize: str = "full",
        month: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Download intraday data from Alpha Vantage.

        Args:
            ticker: Stock ticker symbol
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            outputsize: 'compact' (last 100 bars) or 'full' (extended history)
            month: Optional month slice (format: 'year-month' e.g. '2024-01')
            force_refresh: Force re-download even if cached

        Returns:
            DataFrame with intraday OHLCV data
        """
        # Check cache first
        cache_key = f"{ticker}_{interval}" + (f"_{month}" if month else "")
        if self.use_cache and not force_refresh:
            cached_data = self._load_from_cache(cache_key, interval)
            if cached_data is not None:
                return cached_data

        # Respect rate limits
        self._wait_for_rate_limit()

        # Prepare request
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': ticker,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': outputsize,
            'datatype': 'json',
        }

        # Add month slice if specified
        if month:
            params['month'] = month
            logger.info(f"  Downloading {ticker} {interval} for {month}...")
        else:
            logger.info(f"  Downloading {ticker} {interval} ({outputsize})...")

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for errors
            if 'Error Message' in data:
                logger.error(f"  ✗ Alpha Vantage error: {data['Error Message']}")
                return pd.DataFrame()

            if 'Note' in data:
                logger.warning(f"  ⚠ Rate limit message: {data['Note']}")
                time.sleep(60)  # Wait a full minute
                return self.download_intraday(ticker, interval, outputsize, month)  # Retry

            if 'Information' in data:
                logger.warning(f"  ⚠ {data['Information']}")
                return pd.DataFrame()

            # Parse time series data
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                logger.error(f"  ✗ Unexpected response format")
                logger.debug(f"Response keys: {data.keys()}")
                return pd.DataFrame()

            time_series = data[time_series_key]

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns (Alpha Vantage format: "1. open", "2. high", etc.)
            df.columns = ['open', 'high', 'low', 'close', 'volume']

            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            logger.info(f"  ✓ Downloaded {len(df)} bars")

            # Cache the data
            if self.use_cache:
                self._save_to_cache(df, cache_key, interval)

            return df

        except requests.exceptions.Timeout:
            logger.error(f"  ✗ Request timeout")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            return pd.DataFrame()

    def download_extended_history(
        self,
        ticker: str,
        interval: str = "5min",
        months_back: int = 6,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Download extended intraday history using month slices.

        Alpha Vantage allows downloading specific months of data.
        This method combines multiple months for extended history.

        Args:
            ticker: Stock ticker symbol
            interval: Time interval
            months_back: Number of months to download (max ~24)
            force_refresh: Force re-download even if cached

        Returns:
            Combined DataFrame with extended history
        """
        # Check if we have the full combined dataset cached
        combined_cache_key = f"{ticker}_extended_{months_back}m"
        if self.use_cache and not force_refresh:
            cached_combined = self._load_from_cache(combined_cache_key, interval)
            if cached_combined is not None:
                logger.info(f"\n✓ Loaded extended history from cache")
                logger.info(f"  Total bars: {len(cached_combined):,}")
                logger.info(f"  Date range: {cached_combined.index[0]} to {cached_combined.index[-1]}")
                return cached_combined

        logger.info(f"\nDownloading {months_back} months of {interval} data for {ticker}")
        logger.info("=" * 70)

        all_data = []
        current_date = datetime.now()

        for i in range(months_back):
            # Calculate month to download
            target_date = current_date - timedelta(days=30 * i)
            month_str = target_date.strftime("%Y-%m")

            # Download this month's data (with individual month caching)
            month_data = self.download_intraday(
                ticker=ticker,
                interval=interval,
                outputsize="full",
                month=month_str,
                force_refresh=force_refresh,
            )

            if not month_data.empty:
                all_data.append(month_data)

            # Small delay between requests (only if not from cache)
            if i < months_back - 1 and not month_data.empty:
                time.sleep(1)

        if not all_data:
            logger.error("No data downloaded!")
            return pd.DataFrame()

        # Combine all months
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()

        # Cache the combined dataset
        if self.use_cache:
            self._save_to_cache(combined, combined_cache_key, interval)

        logger.info("\n" + "=" * 70)
        logger.info(f"✓ Extended history complete:")
        logger.info(f"  Total bars: {len(combined):,}")
        logger.info(f"  Date range: {combined.index[0]} to {combined.index[-1]}")
        logger.info(f"  Duration: {(combined.index[-1] - combined.index[0]).days} days")

        return combined


def print_alphavantage_info():
    """Print information about Alpha Vantage API."""
    info = """
╔═══════════════════════════════════════════════════════════════════╗
║                    ALPHA VANTAGE INTRADAY DATA                    ║
╚═══════════════════════════════════════════════════════════════════╝

📊 Data Available:
   • Intraday intervals: 1min, 5min, 15min, 30min, 60min
   • Historical depth: Up to 2 years with extended API
   • Data quality: Real-time, high-quality OHLCV bars
   • Coverage: US stocks, ETFs, and more

🔑 API Key (FREE):
   1. Visit: https://www.alphavantage.co/support/#api-key
   2. Enter your email
   3. Receive API key instantly
   4. No credit card required!

📈 Free Tier Limits:
   • 5 API calls per minute
   • 500 API calls per day
   • Full historical data available (with slicing)

💡 Tips for Maximum Data:
   • Use caching to avoid re-downloads
   • Download extended history with month slicing
   • 5min interval = good balance (2-3 months per call)
   • 60min interval = maximum history (2 years per call)

🚀 Example Usage:
   ```python
   from src.timeseries import AlphaVantageLoader

   # Initialize with your API key
   loader = AlphaVantageLoader(
       api_key="YOUR_API_KEY_HERE",
       use_cache=True  # Highly recommended!
   )

   # Download 6 months of 5-minute data
   data = loader.download_extended_history(
       ticker="AAPL",
       interval="5min",
       months_back=6  # ~75,000 bars!
   )
   ```

📌 Best Intervals for LLM Training:
   • 5min: ~30,000 bars/month → Good for short-term patterns
   • 15min: ~10,000 bars/month → Balance of detail and volume
   • 60min: ~2,500 bars/month → Maximum history (2 years)

⚠️  Important Notes:
   • Free tier: 500 calls/day = ~100 months of data
   • Use caching to maximize API calls
   • Premium tier available for higher limits
   • Data updates in real-time during market hours
    """
    print(info)


if __name__ == "__main__":
    print_alphavantage_info()
