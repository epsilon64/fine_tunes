"""Time series dataset loaders"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataLoader:
    """
    Loader for time series data from various sources.
    """

    def __init__(
        self,
        sequence_length: int = 30,
        prediction_horizon: int = 1,
    ):
        """
        Initialize time series data loader.

        Args:
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps ahead to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

    def load_csv(
        self,
        file_path: str,
        date_column: Optional[str] = None,
        value_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load time series from CSV file.

        Args:
            file_path: Path to CSV file
            date_column: Name of date column (if any)
            value_columns: Columns to use as features

        Returns:
            DataFrame with time series data
        """
        logger.info(f"Loading CSV from: {file_path}")

        df = pd.read_csv(file_path)

        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
            df.set_index(date_column, inplace=True)

        if value_columns:
            df = df[value_columns]

        logger.info(f"Loaded {len(df)} rows")

        return df

    def load_multiple_series(
        self,
        file_paths: List[str],
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple time series files.

        Args:
            file_paths: List of file paths
            **kwargs: Additional arguments for load_csv

        Returns:
            Dictionary mapping file names to DataFrames
        """
        series_dict = {}

        for path in file_paths:
            name = path.split('/')[-1].replace('.csv', '')
            series_dict[name] = self.load_csv(path, **kwargs)

        logger.info(f"Loaded {len(series_dict)} time series")

        return series_dict

    def create_sliding_windows(
        self,
        data: np.ndarray,
        window_size: Optional[int] = None,
        horizon: Optional[int] = None,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences.

        Args:
            data: Time series data (N, features)
            window_size: Size of sliding window
            horizon: Prediction horizon
            stride: Stride for sliding window

        Returns:
            Tuple of (X, y) arrays
        """
        window_size = window_size or self.sequence_length
        horizon = horizon or self.prediction_horizon

        X, y = [], []

        for i in range(0, len(data) - window_size - horizon + 1, stride):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size:i + window_size + horizon])

        return np.array(X), np.array(y)

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        normalize: bool = True,
        fill_na: str = "ffill",
    ) -> Dict[str, np.ndarray]:
        """
        Prepare time series data for training.

        Args:
            df: DataFrame with time series
            train_ratio: Ratio of data for training
            normalize: Whether to normalize data
            fill_na: Method to fill NaN values ('ffill', 'bfill', 'mean', 'zero')

        Returns:
            Dictionary with prepared data
        """
        logger.info("Preparing data for training...")

        # Handle missing values
        if fill_na == "ffill":
            df = df.fillna(method='ffill')
        elif fill_na == "bfill":
            df = df.fillna(method='bfill')
        elif fill_na == "mean":
            df = df.fillna(df.mean())
        elif fill_na == "zero":
            df = df.fillna(0)

        # Convert to numpy
        data = df.values

        # Normalize if requested
        norm_params = None
        if normalize:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0) + 1e-8
            data = (data - mean) / std
            norm_params = {"mean": mean, "std": std}
            logger.info("Data normalized")

        # Create sequences
        X, y = self.create_sliding_windows(data)

        # Train/test split
        split_idx = int(len(X) * train_ratio)

        result = {
            "X_train": X[:split_idx],
            "y_train": y[:split_idx],
            "X_test": X[split_idx:],
            "y_test": y[split_idx:],
            "norm_params": norm_params,
            "column_names": df.columns.tolist(),
        }

        logger.info(
            f"Data prepared: train={len(X[:split_idx])}, "
            f"test={len(X[split_idx:])}, features={data.shape[1]}"
        )

        return result


class MultiVariateTimeSeriesLoader(TimeSeriesDataLoader):
    """
    Specialized loader for multivariate time series.
    """

    def __init__(
        self,
        sequence_length: int = 30,
        prediction_horizon: int = 1,
        target_column: Optional[str] = None,
    ):
        """
        Initialize multivariate loader.

        Args:
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps ahead
            target_column: Specific column to predict (if None, predicts all)
        """
        super().__init__(sequence_length, prediction_horizon)
        self.target_column = target_column

    def prepare_multivariate(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
    ) -> Dict[str, np.ndarray]:
        """
        Prepare multivariate time series.

        Args:
            df: DataFrame with multiple columns
            train_ratio: Train/test split ratio

        Returns:
            Prepared data dictionary
        """
        logger.info(f"Preparing multivariate data with {len(df.columns)} features")

        # Separate features and target if specified
        if self.target_column and self.target_column in df.columns:
            feature_cols = [col for col in df.columns if col != self.target_column]
            features = df[feature_cols].values
            target = df[self.target_column].values.reshape(-1, 1)

            # Create sequences
            X_features, _ = self.create_sliding_windows(features)
            _, y_target = self.create_sliding_windows(target)

            # Split
            split_idx = int(len(X_features) * train_ratio)

            result = {
                "X_train": X_features[:split_idx],
                "y_train": y_target[:split_idx],
                "X_test": X_features[split_idx:],
                "y_test": y_target[split_idx:],
                "feature_columns": feature_cols,
                "target_column": self.target_column,
            }
        else:
            # All features are both input and output
            result = self.prepare_for_training(df, train_ratio)

        return result
