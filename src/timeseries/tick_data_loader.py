"""
Tick data loader supporting multiple free and low-cost data providers.

Supported providers:
- Yahoo Finance (free, limited granularity)
- Alpha Vantage (free tier: 5 calls/min, 500 calls/day)
- Twelve Data (free tier: 800 calls/day)
- Polygon.io (free tier with delayed data)
- IEX Cloud (free tier available)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Literal
import requests
import time
from datetime import datetime, timedelta
import logging
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TickDataLoader:
    """
    Universal tick data loader supporting multiple providers.

    Each provider has different rate limits and data granularity.
    """

    def __init__(
        self,
        provider: str = "yahoo",
        api_key: Optional[str] = None,
    ):
        """
        Initialize tick data loader.

        Args:
            provider: Data provider ('yahoo', 'alphavantage', 'twelvedata', 'polygon', 'iex')
            api_key: API key for the provider (not needed for Yahoo)
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self._validate_provider()

    def _validate_provider(self):
        """Validate provider and API key."""
        supported_providers = ['yahoo', 'alphavantage', 'twelvedata', 'polygon', 'iex']

        if self.provider not in supported_providers:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Supported: {', '.join(supported_providers)}"
            )

        if self.provider != 'yahoo' and not self.api_key:
            logger.warning(
                f"API key not provided for {self.provider}. "
                "Some features may not work."
            )

    def load_intraday_data(
        self,
        ticker: str,
        interval: str = "1min",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1d",
    ) -> pd.DataFrame:
        """
        Load intraday tick data.

        Args:
            ticker: Stock ticker symbol
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period if dates not specified

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Loading {interval} data for {ticker} from {self.provider}")

        if self.provider == 'yahoo':
            return self._load_yahoo_intraday(ticker, interval, period)
        elif self.provider == 'alphavantage':
            return self._load_alphavantage_intraday(ticker, interval)
        elif self.provider == 'twelvedata':
            return self._load_twelvedata_intraday(ticker, interval, start_date, end_date)
        elif self.provider == 'polygon':
            return self._load_polygon_intraday(ticker, interval, start_date, end_date)
        elif self.provider == 'iex':
            return self._load_iex_intraday(ticker)
        else:
            raise ValueError(f"Provider {self.provider} not implemented")

    def _load_yahoo_intraday(
        self,
        ticker: str,
        interval: str,
        period: str,
    ) -> pd.DataFrame:
        """
        Load intraday data from Yahoo Finance.

        Note: Yahoo Finance free tier has limitations on historical intraday data.
        """
        # Map interval formats
        interval_map = {
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m',
            '60min': '60m',
            '1h': '1h',
        }

        yf_interval = interval_map.get(interval, interval)

        try:
            data = yf.download(
                ticker,
                period=period,
                interval=yf_interval,
                progress=False,
                auto_adjust=False,
            )

            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Ensure proper column names
            data.columns = data.columns.str.lower()

            logger.info(f"Loaded {len(data)} records from Yahoo Finance")
            return data

        except Exception as e:
            logger.error(f"Error loading from Yahoo Finance: {e}")
            return pd.DataFrame()

    def _load_alphavantage_intraday(
        self,
        ticker: str,
        interval: str,
    ) -> pd.DataFrame:
        """
        Load intraday data from Alpha Vantage.

        Free tier: 5 API calls/min, 500 calls/day
        """
        if not self.api_key:
            raise ValueError("API key required for Alpha Vantage")

        # Map interval
        interval_map = {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '60min': '60min',
        }

        av_interval = interval_map.get(interval, '5min')

        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': ticker,
            'interval': av_interval,
            'apikey': self.api_key,
            'outputsize': 'full',
            'datatype': 'json',
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Check for errors
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return pd.DataFrame()

            if 'Note' in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return pd.DataFrame()

            # Parse time series data
            time_series_key = f'Time Series ({av_interval})'
            if time_series_key not in data:
                logger.error(f"Unexpected response format from Alpha Vantage")
                return pd.DataFrame()

            time_series = data[time_series_key]

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)

            logger.info(f"Loaded {len(df)} records from Alpha Vantage")
            return df

        except Exception as e:
            logger.error(f"Error loading from Alpha Vantage: {e}")
            return pd.DataFrame()

    def _load_twelvedata_intraday(
        self,
        ticker: str,
        interval: str,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """
        Load intraday data from Twelve Data.

        Free tier: 800 API calls/day
        """
        if not self.api_key:
            raise ValueError("API key required for Twelve Data")

        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': ticker,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': 5000,
            'format': 'JSON',
        }

        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'status' in data and data['status'] == 'error':
                logger.error(f"Twelve Data error: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()

            if 'values' not in data:
                logger.error("Unexpected response format from Twelve Data")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            df = df.sort_index()

            # Rename and convert columns
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)

            logger.info(f"Loaded {len(df)} records from Twelve Data")
            return df

        except Exception as e:
            logger.error(f"Error loading from Twelve Data: {e}")
            return pd.DataFrame()

    def _load_polygon_intraday(
        self,
        ticker: str,
        interval: str,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """
        Load intraday data from Polygon.io.

        Free tier: Delayed data (15 minutes)
        """
        if not self.api_key:
            raise ValueError("API key required for Polygon.io")

        # Parse interval
        interval_parts = interval.replace('min', '').replace('h', '')
        try:
            multiplier = int(interval_parts)
            timespan = 'minute' if 'min' in interval else 'hour'
        except:
            multiplier = 1
            timespan = 'minute'

        # Set default dates
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {
            'apiKey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'OK':
                logger.error(f"Polygon.io error: {data.get('error', 'Unknown error')}")
                return pd.DataFrame()

            if 'results' not in data or not data['results']:
                logger.warning(f"No data returned from Polygon.io")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            df['datetime'] = pd.to_datetime(df['t'], unit='ms')
            df = df.set_index('datetime')
            df = df.sort_index()

            # Rename columns: v=volume, vw=vwap, o=open, c=close, h=high, l=low
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
            })

            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.astype(float)

            logger.info(f"Loaded {len(df)} records from Polygon.io")
            return df

        except Exception as e:
            logger.error(f"Error loading from Polygon.io: {e}")
            return pd.DataFrame()

    def _load_iex_intraday(
        self,
        ticker: str,
    ) -> pd.DataFrame:
        """
        Load intraday data from IEX Cloud.

        Free tier available with limitations.
        """
        if not self.api_key:
            raise ValueError("API key required for IEX Cloud")

        url = f"https://cloud.iexapis.com/stable/stock/{ticker}/intraday-prices"
        params = {
            'token': self.api_key,
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                logger.warning(f"No data returned from IEX Cloud")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Create datetime index
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['minute'])
            df = df.set_index('datetime')
            df = df.sort_index()

            # Select and rename columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.astype(float)

            logger.info(f"Loaded {len(df)} records from IEX Cloud")
            return df

        except Exception as e:
            logger.error(f"Error loading from IEX Cloud: {e}")
            return pd.DataFrame()

    def get_tick_statistics(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate tick-level statistics.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with tick statistics
        """
        if data.empty:
            return {}

        stats = {
            'total_ticks': len(data),
            'start_time': data.index[0],
            'end_time': data.index[-1],
            'duration_hours': (data.index[-1] - data.index[0]).total_seconds() / 3600,

            # Price statistics
            'mean_price': float(data['close'].mean()),
            'price_std': float(data['close'].std()),
            'price_range': float(data['high'].max() - data['low'].min()),
            'total_return': float((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]),

            # Volume statistics
            'total_volume': int(data['volume'].sum()),
            'mean_volume': float(data['volume'].mean()),
            'volume_std': float(data['volume'].std()),

            # Volatility
            'tick_returns': data['close'].pct_change().dropna(),
            'volatility': float(data['close'].pct_change().std()),
            'max_drawdown': self._calculate_max_drawdown(data['close'].values),
        }

        return stats

    @staticmethod
    def _calculate_max_drawdown(prices: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        return float(drawdown.min())

    def resample_ticks(
        self,
        data: pd.DataFrame,
        freq: str,
    ) -> pd.DataFrame:
        """
        Resample tick data to different frequency.

        Args:
            data: DataFrame with OHLCV data
            freq: Resampling frequency ('5min', '15min', '1h', etc.)

        Returns:
            Resampled DataFrame
        """
        if data.empty:
            return data

        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        })

        # Drop rows with NaN (no data in that period)
        resampled = resampled.dropna()

        logger.info(f"Resampled from {len(data)} to {len(resampled)} bars")
        return resampled


def get_provider_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about supported data providers.

    Returns:
        Dictionary with provider information
    """
    return {
        'yahoo': {
            'name': 'Yahoo Finance',
            'cost': 'Free',
            'api_key_required': 'No',
            'rate_limit': 'Unknown (liberal)',
            'historical_limit': 'Limited intraday history (7 days)',
            'intervals': '1m, 5m, 15m, 30m, 60m',
            'signup_url': 'None (no registration needed)',
            'notes': 'Best for getting started, limited historical intraday data',
        },
        'alphavantage': {
            'name': 'Alpha Vantage',
            'cost': 'Free tier available',
            'api_key_required': 'Yes',
            'rate_limit': '5 calls/min, 500 calls/day',
            'historical_limit': 'Full historical intraday data',
            'intervals': '1min, 5min, 15min, 30min, 60min',
            'signup_url': 'https://www.alphavantage.co/support/#api-key',
            'notes': 'Good for research, rate limits apply',
        },
        'twelvedata': {
            'name': 'Twelve Data',
            'cost': 'Free tier: 800 calls/day',
            'api_key_required': 'Yes',
            'rate_limit': '8 calls/min, 800 calls/day (free)',
            'historical_limit': 'Up to 5000 data points per call',
            'intervals': '1min, 5min, 15min, 30min, 1h, etc.',
            'signup_url': 'https://twelvedata.com/pricing',
            'notes': 'Good balance of features and limits',
        },
        'polygon': {
            'name': 'Polygon.io',
            'cost': 'Free tier with delayed data',
            'api_key_required': 'Yes',
            'rate_limit': '5 calls/min (free tier)',
            'historical_limit': '2 years historical data',
            'intervals': 'Flexible (1min to 1day)',
            'signup_url': 'https://polygon.io/pricing',
            'notes': 'Free tier has 15-minute delay',
        },
        'iex': {
            'name': 'IEX Cloud',
            'cost': 'Free tier available',
            'api_key_required': 'Yes',
            'rate_limit': 'Based on credits',
            'historical_limit': 'Varies by plan',
            'intervals': 'Intraday minutes',
            'signup_url': 'https://iexcloud.io/pricing',
            'notes': 'Credit-based system',
        },
    }


def print_provider_comparison():
    """Print comparison table of data providers."""
    providers = get_provider_info()

    print("\n" + "=" * 100)
    print("TICK DATA PROVIDER COMPARISON")
    print("=" * 100)

    for provider, info in providers.items():
        print(f"\n{info['name'].upper()} ({provider})")
        print("-" * 100)
        print(f"  Cost:               {info['cost']}")
        print(f"  API Key Required:   {info['api_key_required']}")
        print(f"  Rate Limit:         {info['rate_limit']}")
        print(f"  Historical Limit:   {info['historical_limit']}")
        print(f"  Intervals:          {info['intervals']}")
        print(f"  Signup URL:         {info['signup_url']}")
        print(f"  Notes:              {info['notes']}")

    print("\n" + "=" * 100)
    print("RECOMMENDATION: Start with 'yahoo' (no API key needed), then upgrade to 'alphavantage'")
    print("                or 'twelvedata' for more historical data and better intervals.")
    print("=" * 100 + "\n")
