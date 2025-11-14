# LLM Fine-Tuning Project

A comprehensive framework for fine-tuning Large Language Models with efficient techniques and specialized time series forecasting capabilities.

## Features

### General LLM Fine-Tuning
- **Model Loading**: Support for various small LLMs (GPT-2, LLaMA, Mistral, etc.)
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning with parameter-efficient training
- **Quantization**: 4-bit and 8-bit quantization support via bitsandbytes
- **Flexible LoRA Application**: Apply LoRA to specific model components (attention, MLP, etc.)
- **Multiple Backend Support**: Compatible with HuggingFace transformers

### Time Series Forecasting
- **Specialized Foundation Models**:
  - **Chronos** (Amazon): T5-based, zero-shot forecasting, 5 model sizes (8M-710M params)
  - **Lag-Llama**: Llama-based, probabilistic forecasting with uncertainty quantification
- **Custom LLM Adaptation**: Adapt GPT-2, LLaMA, Mistral for time series
- **Financial Time Series**: Specialized module for stock market price prediction
- **Return Prediction**: Predict future returns based on historical prices/returns
- **Tick Data Support**: Load high-frequency intraday data from multiple providers
  - Yahoo Finance (free, no API key needed)
  - Alpha Vantage (free tier: 500 calls/day)
  - Twelve Data (free tier: 800 calls/day)
  - Polygon.io (free tier with delayed data)
  - IEX Cloud (credit-based free tier)
- **Comprehensive Error Metrics**: MSE, RMSE, MAE, MAPE, R², Direction Accuracy, and more
- **Advanced Bar Types**: Time bars, volume bars, and tick bars
- **Tick-specific Features**: VWAP, spread proxy, volume intensity, intraday volatility

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Fine-Tuning with LoRA

```python
from src.core.model_loader import ModelLoader
from src.core.lora import LoRAConfig, apply_lora
from src.core.trainer import FineTuner

# Load model
loader = ModelLoader()
model = loader.load_model("gpt2", quantization="4bit")

# Configure LoRA
lora_config = LoRAConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

# Apply LoRA
model = apply_lora(model, lora_config)

# Fine-tune
trainer = FineTuner(model, tokenizer)
trainer.train(train_dataset)
```

### Time Series Forecasting

```python
from src.timeseries.ts_model import TimeSeriesLLM
from src.timeseries.financial_preprocessor import FinancialDataPreprocessor

# Prepare data
preprocessor = FinancialDataPreprocessor()
data = preprocessor.prepare_data(ticker="AAPL", start_date="2020-01-01")

# Create and train model
ts_model = TimeSeriesLLM(
    model_name="gpt2",
    d_input=data['X_train'].shape[-1],
    d_output=data['y_train'].shape[-1]
)

from src.timeseries.ts_trainer import TimeSeriesTrainer
trainer = TimeSeriesTrainer(ts_model)
trainer.train(data, epochs=10)

# Predict
predictions = trainer.predict(data["X_test"], steps_ahead=5)
```

### Alpha Vantage Intraday Data (TRUE Tick-Level Data)

For true intraday/tick-level data with extended history:

```python
from src.timeseries import AlphaVantageLoader

# Get FREE API key: https://www.alphavantage.co/support/#api-key
loader = AlphaVantageLoader(
    api_key="YOUR_FREE_API_KEY",
    cache_dir="./alphavantage_cache",
    use_cache=True  # Avoid re-downloads!
)

# Download 6 months of 5-minute bars (~75,000 bars!)
intraday_data = loader.download_extended_history(
    ticker="AAPL",
    interval="5min",  # 1min, 5min, 15min, 30min, 60min
    months_back=6,
)
```

**Alpha Vantage vs Yahoo Finance:**

| Feature | Alpha Vantage | Yahoo Finance |
|---------|---------------|---------------|
| **Intraday Resolution** | 1min, 5min, 15min, 30min, 60min | 1min, 5min, 15min, 30min, 60min |
| **Historical Depth** | Up to 2 years | 7-60 days |
| **Data Quality** | Professional-grade | Consumer-grade |
| **Free Tier** | 5 calls/min, 500/day | Unlimited but limited history |
| **Caching** | Built-in | Not needed (limited data) |
| **Best For** | LLM training with deep history | Quick experiments |

**Data Volume Comparison:**
- Alpha Vantage 5min × 6 months = **~75,000 bars** ✅
- Yahoo Finance 5min × 7 days = **~2,000 bars** ❌

Run the example:
```bash
# Set your API key
export ALPHAVANTAGE_API_KEY='your_key_here'

# Run example
python examples/alphavantage_intraday.py
```

### Tick Data Loading (Intraday Data)

```python
from src.timeseries.tick_data_loader import TickDataLoader

# Load intraday data (no API key needed for Yahoo)
tick_loader = TickDataLoader(provider="yahoo")
tick_data = tick_loader.load_intraday_data(
    ticker="AAPL",
    interval="5min",
    period="5d"
)

# Or use paid providers for more data
tick_loader = TickDataLoader(provider="alphavantage", api_key="YOUR_API_KEY")
tick_data = tick_loader.load_intraday_data(ticker="AAPL", interval="1min")

# Prepare tick data for training
preprocessor = FinancialDataPreprocessor()
data = preprocessor.prepare_tick_data(tick_data)
```

### Specialized Time Series Models (Zero-Shot!)

```python
from src.timeseries import ChronosModel

# Load Chronos model (works without training!)
model = ChronosModel(
    model_size="small",  # tiny, mini, small, base, large
    device="cpu"
)

# Generate zero-shot forecasts
forecast = model.predict(
    context=historical_data,
    prediction_length=5,
    num_samples=20  # For uncertainty quantification
)

# Get median prediction
prediction = forecast.median(dim=0).values

# See SPECIALIZED_MODELS.md for full guide
```

### Visualizing Forecasts

```python
from src.timeseries import ForecastVisualizer

# Create visualizer
visualizer = ForecastVisualizer(output_dir="./outputs")

# 1. Comprehensive comparison (4-panel analysis)
visualizer.plot_comprehensive_comparison(
    predictions=predictions,
    actuals=actuals,
    title="Forecast Analysis"
)
# Creates: time series, scatter plot, error distribution, cumulative error

# 2. Price reconstruction from returns
visualizer.plot_price_reconstruction(
    returns_predictions=returns_pred,
    returns_actuals=returns_actual,
    initial_price=100.0,  # Starting price
    title="Price Forecast vs Realized Prices"
)
# Shows: predicted vs actual price paths, cumulative returns

# 3. Multi-horizon accuracy analysis
visualizer.plot_forecast_horizon_analysis(
    predictions=predictions,  # Shape: (n_samples, n_horizons, n_features)
    actuals=actuals,
    horizon_names=["H+1", "H+2", "H+3", "H+5"],
    title="Multi-Step Forecast Analysis"
)
# Shows: MAE, RMSE, Direction Accuracy across horizons

# 4. Single time series comparison with metrics
visualizer.plot_predictions_vs_actual(
    predictions=predictions,
    actuals=actuals,
    dates=date_index,  # Optional
    show_confidence=True,  # Show confidence intervals
    confidence_lower=lower_bound,
    confidence_upper=upper_bound
)
```

**Key Visualization Features:**
- **Price Reconstruction**: Convert return predictions back to prices
- **Comprehensive Metrics**: MAE, RMSE, MAPE, R², Direction Accuracy
- **Multi-Horizon Analysis**: Track accuracy degradation across forecast steps
- **Confidence Intervals**: Visualize prediction uncertainty
- **High-Quality Output**: 300 DPI PNG files, professional styling

All examples now include these visualizations automatically!

### Multi-Ticker Training with Maximum Historical Data

Train on up to 15 years of data from 30+ tickers for best LLM performance:

```python
from src.timeseries import BulkDataLoader, get_recommended_tickers

# Download maximum historical data (cached for fast re-runs)
tickers = get_recommended_tickers('sp500_top')[:30]  # Top 30 S&P 500
loader = BulkDataLoader(cache_dir="./data_cache", use_cache=True)

# Pull 15 years of daily data per ticker
ticker_data = loader.download_multiple_tickers(
    tickers=tickers,
    start_date="2010-01-01",  # ~15 years of history
    delay=0.3,  # Respectful rate limiting
)

# This generates 100,000+ training samples!
# LLMs need substantial data for good performance
```

**Data Scale for LLM Training:**
- **Minimum**: 10,000 samples for basic performance
- **Recommended**: 50,000+ samples for good results
- **This Example**: 100,000+ samples from 30 tickers × 15 years
- **Total Data Points**: 3,000,000+ individual observations

**Benefits:**
- ✅ **Maximum Data**: Up to 15 years per ticker (not limited to 1 year)
- ✅ **True Historical Data**: Daily OHLCV data (not limited "tick" data)
- ✅ **Smart Caching**: Saves downloads to avoid hitting rate limits
- ✅ **Better Generalization**: Learns from bull, bear, and sideways markets
- ✅ **Improved Accuracy**: Vastly more diverse training data
- ✅ **Transfer Learning**: Apply to new stocks without retraining
- ✅ **Market Regimes**: Captures different volatility periods

**Recommended Ticker Sets:**
```python
# Tech-focused (20 stocks)
tickers = get_recommended_tickers('tech')

# Diverse sectors (20 stocks)
tickers = get_recommended_tickers('diverse')

# Top S&P 500 (30 stocks)
tickers = get_recommended_tickers('sp500_top')
```

Run the example:
```bash
python examples/multi_ticker_training.py
# Downloads 30 tickers × ~3,500 days = 100,000+ samples
# Training time: ~30-60 minutes on CPU
```

## Project Structure

```
├── src/
│   ├── core/          # Core fine-tuning modules
│   ├── datasets/      # Dataset loaders
│   └── timeseries/    # Time series specific modules
├── examples/          # Example scripts
├── notebooks/         # Jupyter notebooks
└── config/           # Configuration files
```

## Configuration

Configuration files in `config/` allow you to customize:
- Model architecture and size
- LoRA parameters (rank, alpha, target modules)
- Quantization settings
- Training hyperparameters
- Time series specific settings

## Examples

See the `examples/` directory for:
- `basic_finetuning.py`: Standard fine-tuning workflow
- `lora_finetuning.py`: LoRA-based efficient fine-tuning
- `timeseries_forecasting.py`: Financial time series prediction
- `timeseries_detailed.py`: Comprehensive forecasting with error analysis
- `tick_data_forecasting.py`: High-frequency tick data forecasting
- `specialized_models_forecasting.py`: Zero-shot forecasting with Chronos/Lag-Llama
- `multi_ticker_training.py`: Train on multiple stocks with 15 years of data
- `alphavantage_intraday.py`: **NEW!** True intraday data with extended history (75,000+ bars)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (Parameter-Efficient Fine-Tuning)
- bitsandbytes (for quantization)
- pandas, numpy (for data processing)

## License

MIT License
