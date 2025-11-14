"""
Configuration loader for the LLM fine-tuning project.

Loads configuration from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import Optional


def load_env():
    """
    Load environment variables from .env file if it exists.

    Uses python-dotenv if available, otherwise falls back to manual parsing.
    """
    env_path = Path(__file__).parent.parent / ".env"

    if not env_path.exists():
        return

    try:
        # Try using python-dotenv (recommended)
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # Fallback: manual parsing
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key and value and not os.getenv(key):
                        os.environ[key] = value


class Config:
    """Configuration singleton for the project."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Load .env file
        load_env()

        # API Keys
        self.alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        self.twelvedata_api_key = os.getenv("TWELVEDATA_API_KEY")
        self.polygon_api_key = os.getenv("POLYGON_API_KEY")
        self.iex_api_key = os.getenv("IEX_API_KEY")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        # Model Configuration
        self.model_name = os.getenv("MODEL_NAME", "gpt2")
        self.device = os.getenv("DEVICE", "cpu")

        # Paths
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "./outputs"))
        self.cache_dir = Path(os.getenv("CACHE_DIR", "./data_cache"))

        # Data Configuration
        self.default_tickers = os.getenv("DEFAULT_TICKERS", "AAPL,GOOGL,MSFT").split(",")
        self.data_start_date = os.getenv("DATA_START_DATE", "2010-01-01")
        self.sequence_length = int(os.getenv("SEQUENCE_LENGTH", "30"))
        self.prediction_horizon = int(os.getenv("PREDICTION_HORIZON", "5"))

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        self._initialized = True

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a specific provider.

        Args:
            provider: Provider name ('alphavantage', 'twelvedata', 'polygon', 'iex')

        Returns:
            API key or None if not set
        """
        provider = provider.lower()
        key_map = {
            'alphavantage': self.alphavantage_api_key,
            'twelvedata': self.twelvedata_api_key,
            'polygon': self.polygon_api_key,
            'iex': self.iex_api_key,
        }
        return key_map.get(provider)

    def __repr__(self):
        """String representation (hides API keys)."""
        masked_keys = {
            'alphavantage': '***' if self.alphavantage_api_key else 'Not set',
            'twelvedata': '***' if self.twelvedata_api_key else 'Not set',
            'polygon': '***' if self.polygon_api_key else 'Not set',
            'iex': '***' if self.iex_api_key else 'Not set',
        }

        return f"""
Config:
  API Keys:
    Alpha Vantage: {masked_keys['alphavantage']}
    Twelve Data: {masked_keys['twelvedata']}
    Polygon.io: {masked_keys['polygon']}
    IEX Cloud: {masked_keys['iex']}

  Model:
    Name: {self.model_name}
    Device: {self.device}

  Paths:
    Output: {self.output_dir}
    Cache: {self.cache_dir}

  Data:
    Tickers: {', '.join(self.default_tickers)}
    Start Date: {self.data_start_date}
    Sequence Length: {self.sequence_length}
    Prediction Horizon: {self.prediction_horizon}

  Logging:
    Level: {self.log_level}
"""


# Singleton instance
config = Config()


if __name__ == "__main__":
    print(config)
