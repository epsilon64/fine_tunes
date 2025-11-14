"""Financial data preprocessing for time series LLMs"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataPreprocessor:
    """
    Preprocessor for financial time series data.

    Handles stock price data, returns calculation, and sequence preparation.
    """

    def __init__(
        self,
        sequence_length: int = 30,
        prediction_horizon: int = 1,
        return_type: str = "log",  # 'log' or 'simple'
    ):
        """
        Initialize preprocessor.

        Args:
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
            return_type: Type of returns ('log' or 'simple')
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.return_type = return_type

    def load_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
    ) -> pd.DataFrame:
        """
        Load stock price data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period if dates not specified (e.g., '1y', '2y', '5y')

        Returns:
            DataFrame with stock data
        """
        logger.info(f"Loading data for {ticker}")

        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        else:
            data = yf.download(ticker, period=period, progress=False)

        logger.info(f"Loaded {len(data)} data points")
        return data

    def calculate_returns(
        self,
        prices: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate returns from prices.

        Args:
            prices: Price series

        Returns:
            Array of returns
        """
        prices = np.array(prices)

        if self.return_type == "log":
            # Log returns: log(P_t / P_{t-1})
            returns = np.log(prices[1:] / prices[:-1])
        else:
            # Simple returns: (P_t - P_{t-1}) / P_{t-1}
            returns = (prices[1:] - prices[:-1]) / prices[:-1]

        return returns

    def calculate_technical_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate technical indicators.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with additional technical features
        """
        df = data.copy()

        # Returns
        df['returns'] = self.calculate_returns(df['Close'].values)

        # Moving averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()

        # Volatility (rolling std)
        df['volatility'] = df['returns'].rolling(window=20).std()

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Volume change
        df['volume_change'] = df['Volume'].pct_change()

        return df

    def normalize_data(
        self,
        data: np.ndarray,
        method: str = "standardize",
        fit_params: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize data.

        Args:
            data: Data to normalize
            method: Normalization method ('standardize', 'minmax', or 'none')
            fit_params: Pre-computed normalization parameters

        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        if method == "none":
            return data, {}

        if fit_params is None:
            if method == "standardize":
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0) + 1e-8
                fit_params = {"mean": mean, "std": std}
            elif method == "minmax":
                min_val = np.min(data, axis=0)
                max_val = np.max(data, axis=0)
                fit_params = {"min": min_val, "max": max_val}
            else:
                raise ValueError(f"Unknown normalization method: {method}")

        if method == "standardize":
            normalized = (data - fit_params["mean"]) / fit_params["std"]
        elif method == "minmax":
            normalized = (data - fit_params["min"]) / (fit_params["max"] - fit_params["min"])

        return normalized, fit_params

    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: Optional[int] = None,
        prediction_horizon: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for training.

        Args:
            data: Time series data (N x features)
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps ahead to predict

        Returns:
            Tuple of (X, y) where X is input sequences and y is targets
        """
        seq_len = sequence_length or self.sequence_length
        pred_horizon = prediction_horizon or self.prediction_horizon

        X, y = [], []

        for i in range(len(data) - seq_len - pred_horizon + 1):
            # Input sequence
            X.append(data[i:i + seq_len])
            # Target (future values)
            y.append(data[i + seq_len:i + seq_len + pred_horizon])

        return np.array(X), np.array(y)

    def prepare_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        features: List[str] = None,
        normalize: str = "standardize",
        train_ratio: float = 0.8,
    ) -> Dict[str, np.ndarray]:
        """
        Complete data preparation pipeline.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            features: List of features to use (if None, uses returns only)
            normalize: Normalization method
            train_ratio: Ratio of data for training

        Returns:
            Dictionary with train/test splits and metadata
        """
        # Load data
        data = self.load_stock_data(ticker, start_date, end_date)

        # Calculate features
        data = self.calculate_technical_features(data)

        # Select features
        if features is None:
            features = ['returns']

        # Drop NaN values
        data = data.dropna()

        # Extract feature matrix
        feature_data = data[features].values

        # Normalize
        feature_data, norm_params = self.normalize_data(feature_data, method=normalize)

        # Create sequences
        X, y = self.create_sequences(feature_data)

        # Train/test split
        split_idx = int(len(X) * train_ratio)

        result = {
            "X_train": X[:split_idx],
            "y_train": y[:split_idx],
            "X_test": X[split_idx:],
            "y_test": y[split_idx:],
            "norm_params": norm_params,
            "features": features,
            "dates": data.index.values,
        }

        logger.info(f"Prepared data: train={len(X[:split_idx])}, test={len(X[split_idx:])}")

        return result

    def inverse_transform(
        self,
        data: np.ndarray,
        norm_params: Dict,
        method: str = "standardize",
    ) -> np.ndarray:
        """
        Inverse transform normalized data.

        Args:
            data: Normalized data
            norm_params: Normalization parameters
            method: Normalization method used

        Returns:
            Original scale data
        """
        if method == "standardize":
            return data * norm_params["std"] + norm_params["mean"]
        elif method == "minmax":
            return data * (norm_params["max"] - norm_params["min"]) + norm_params["min"]
        else:
            return data

    def prepare_tick_data(
        self,
        tick_data: pd.DataFrame,
        features: Optional[List[str]] = None,
        normalize: str = "standardize",
        train_ratio: float = 0.8,
    ) -> Dict[str, np.ndarray]:
        """
        Prepare tick/intraday data for training.

        Args:
            tick_data: DataFrame with tick data (from TickDataLoader)
            features: List of features to use
            normalize: Normalization method
            train_ratio: Train/test split ratio

        Returns:
            Dictionary with prepared data
        """
        logger.info("Preparing tick data for training...")

        # Calculate returns and features for tick data
        tick_data = self._calculate_tick_features(tick_data)

        # Select features
        if features is None:
            features = ['tick_returns']

        # Drop NaN values
        tick_data = tick_data.dropna()

        # Extract feature matrix
        feature_data = tick_data[features].values

        # Normalize
        feature_data, norm_params = self.normalize_data(feature_data, method=normalize)

        # Create sequences
        X, y = self.create_sequences(feature_data)

        # Train/test split
        split_idx = int(len(X) * train_ratio)

        result = {
            "X_train": X[:split_idx],
            "y_train": y[:split_idx],
            "X_test": X[split_idx:],
            "y_test": y[split_idx:],
            "norm_params": norm_params,
            "features": features,
            "dates": tick_data.index.values,
        }

        logger.info(f"Tick data prepared: train={len(X[:split_idx])}, test={len(X[split_idx:])}")

        return result

    def _calculate_tick_features(
        self,
        tick_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate features specific to tick data.

        Args:
            tick_data: DataFrame with OHLCV tick data

        Returns:
            DataFrame with tick features
        """
        df = tick_data.copy()

        # Tick returns
        df['tick_returns'] = self.calculate_returns(df['close'].values)

        # Intraday volatility (high-low range)
        df['hl_range'] = (df['high'] - df['low']) / df['close']

        # Bid-ask spread proxy (high - low as percentage of close)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']

        # Volume intensity (volume relative to rolling average)
        df['volume_intensity'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)

        # Price momentum (short-term moving average)
        df['momentum_5'] = df['close'].rolling(5).mean() / df['close']
        df['momentum_10'] = df['close'].rolling(10).mean() / df['close']

        # Intraday VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-8)
        df['vwap_deviation'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-8)

        # Tick volatility (rolling std of returns)
        df['tick_volatility'] = df['tick_returns'].rolling(20).std()

        return df

    def aggregate_ticks_to_bars(
        self,
        tick_data: pd.DataFrame,
        bar_type: str = "time",
        bar_size: Union[int, str] = "5min",
    ) -> pd.DataFrame:
        """
        Aggregate tick data into bars.

        Args:
            tick_data: DataFrame with tick data
            bar_type: Type of bars ('time', 'volume', 'tick')
            bar_size: Size of bars (for time: '5min', '1h'; for volume/tick: integer)

        Returns:
            DataFrame with aggregated bars
        """
        if bar_type == "time":
            # Time-based bars (standard OHLCV resampling)
            bars = tick_data.resample(bar_size).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }).dropna()

        elif bar_type == "volume":
            # Volume bars: each bar has approximately equal volume
            bars = self._create_volume_bars(tick_data, int(bar_size))

        elif bar_type == "tick":
            # Tick bars: each bar has fixed number of ticks
            bars = self._create_tick_bars(tick_data, int(bar_size))

        else:
            raise ValueError(f"Unknown bar type: {bar_type}")

        logger.info(f"Created {len(bars)} {bar_type} bars from {len(tick_data)} ticks")
        return bars

    def _create_volume_bars(
        self,
        tick_data: pd.DataFrame,
        volume_per_bar: int,
    ) -> pd.DataFrame:
        """Create volume-based bars."""
        bars = []
        current_volume = 0
        current_bar = {
            'open': None,
            'high': -np.inf,
            'low': np.inf,
            'close': None,
            'volume': 0,
            'timestamp': None,
        }

        for idx, row in tick_data.iterrows():
            if current_bar['open'] is None:
                current_bar['open'] = row['open']
                current_bar['timestamp'] = idx

            current_bar['high'] = max(current_bar['high'], row['high'])
            current_bar['low'] = min(current_bar['low'], row['low'])
            current_bar['close'] = row['close']
            current_bar['volume'] += row['volume']

            current_volume += row['volume']

            if current_volume >= volume_per_bar:
                bars.append(current_bar.copy())
                current_volume = 0
                current_bar = {
                    'open': None,
                    'high': -np.inf,
                    'low': np.inf,
                    'close': None,
                    'volume': 0,
                    'timestamp': None,
                }

        if current_bar['open'] is not None:
            bars.append(current_bar)

        df = pd.DataFrame(bars)
        df = df.set_index('timestamp')
        return df[['open', 'high', 'low', 'close', 'volume']]

    def _create_tick_bars(
        self,
        tick_data: pd.DataFrame,
        ticks_per_bar: int,
    ) -> pd.DataFrame:
        """Create tick-count based bars."""
        bars = []

        for i in range(0, len(tick_data), ticks_per_bar):
            chunk = tick_data.iloc[i:i + ticks_per_bar]
            if len(chunk) == 0:
                continue

            bar = {
                'open': chunk['open'].iloc[0],
                'high': chunk['high'].max(),
                'low': chunk['low'].min(),
                'close': chunk['close'].iloc[-1],
                'volume': chunk['volume'].sum(),
            }
            bars.append(bar)

        df = pd.DataFrame(bars)
        # Use the last timestamp of each bar
        timestamps = [tick_data.index[min(i + ticks_per_bar - 1, len(tick_data) - 1)]
                     for i in range(0, len(tick_data), ticks_per_bar)]
        df.index = timestamps[:len(df)]

        return df
