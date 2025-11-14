# Testing and Running Examples

This guide explains how to test and run all the examples in the LLM fine-tuning framework.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Quick Tests

The fastest way to verify everything works:

```bash
python tests/quick_test_all.py
```

This runs 6 tests in ~3-5 minutes:
- ✓ Basic imports
- ✓ Data loading (daily stock data)
- ✓ Tick data loading (intraday)
- ✓ Model creation
- ✓ End-to-end training (1 epoch)
- ✓ Error metrics calculation

## Running Examples

All examples are in the `examples/` directory. They're designed to work out of the box.

### Example 1: Basic Time Series Forecasting

**File:** `examples/timeseries_forecasting.py`

Basic stock price prediction with LLM:

```bash
python examples/timeseries_forecasting.py
```

**What it does:**
- Loads AAPL stock data (2 years)
- Calculates returns
- Trains TimeSeriesLLM with LoRA
- Generates predictions
- Creates visualizations

**Output:**
- Training plot
- Predictions plot
- Strategy performance

**Runtime:** ~5-10 minutes

---

### Example 2: Detailed Error Analysis

**File:** `examples/timeseries_detailed.py`

Comprehensive forecasting with 15+ error metrics:

```bash
python examples/timeseries_detailed.py
```

**What it does:**
- Multi-feature training (returns, SMA, RSI, volatility)
- Comprehensive error analysis
- Rolling window metrics
- Trading strategy simulation

**Error Metrics Calculated:**
- MSE, RMSE, MAE, MAPE
- Median AE, Median APE
- R², Direction Accuracy
- Bias, Error Std
- Percentiles (50th, 90th, 95th, 99th)

**Output Files:**
- `outputs/timeseries_detailed/error_analysis.png` (6-panel analysis)
- `outputs/timeseries_detailed/rolling_metrics.png`
- `outputs/timeseries_detailed/strategy_performance.png`
- `outputs/timeseries_detailed/forecast_metrics.csv`
- `outputs/timeseries_detailed/predictions.csv`

**Runtime:** ~10-15 minutes

---

### Example 3: Tick Data Forecasting

**File:** `examples/tick_data_forecasting.py`

High-frequency intraday prediction:

```bash
python examples/tick_data_forecasting.py
```

**What it does:**
- Loads 5-minute intraday data (Yahoo Finance - free!)
- Calculates tick-specific features (VWAP, spread, momentum)
- Trains on high-frequency data
- Generates intraday predictions

**Features:**
- Tick returns
- High-low range
- Volume intensity
- VWAP deviation
- Intraday momentum
- Tick volatility

**Output:**
- `outputs/tick_forecasting/tick_data_raw.png`
- `outputs/tick_forecasting/tick_forecasting_results.png`
- `outputs/tick_forecasting/trading_strategy.png`
- `outputs/tick_forecasting/tick_predictions.csv`

**Runtime:** ~5-10 minutes

---

## Using Different Data Providers

### Yahoo Finance (Free, No API Key)

```python
from src.timeseries import TickDataLoader

loader = TickDataLoader(provider="yahoo")
data = loader.load_intraday_data(
    ticker="AAPL",
    interval="5min",  # 1min, 5min, 15min, 30min, 60min
    period="5d"       # 1d, 5d, 1mo, 3mo
)
```

**Limitations:**
- Limited historical intraday data (7 days max for 1min)
- More history available for longer intervals

---

### Alpha Vantage (Free Tier: 500 calls/day)

**Get Free API Key:** https://www.alphavantage.co/support/#api-key

```python
loader = TickDataLoader(provider="alphavantage", api_key="YOUR_KEY")
data = loader.load_intraday_data(
    ticker="AAPL",
    interval="1min"  # 1min, 5min, 15min, 30min, 60min
)
```

**Advantages:**
- Full historical intraday data
- Reliable 1-minute data
- Good for research

---

### Twelve Data (Free Tier: 800 calls/day)

**Get Free API Key:** https://twelvedata.com/pricing

```python
loader = TickDataLoader(provider="twelvedata", api_key="YOUR_KEY")
data = loader.load_intraday_data(
    ticker="AAPL",
    interval="1min",
    start_date="2024-01-01",
    end_date="2024-01-31"
)
```

**Advantages:**
- More generous free tier (800 calls/day)
- Good balance of features

---

## Troubleshooting

### Error: "Length mismatch"

**Fixed!** This was caused by returns calculation. Now using pandas methods that properly handle NaN padding.

### Error: "tuple object has no attribute 'lower'"

**Fixed!** This was caused by yfinance returning MultiIndex columns. Now automatically detects and flattens them.

### Error: "Missing 'Close' column"

Check your data source. The error message now shows available columns:
```
ValueError: Missing 'Close' column. Available columns: ['close', 'high', 'low', 'volume']
```

Solution: Use a different data provider or check ticker symbol.

### Warning: "YF.download() has changed auto_adjust default"

**Fixed!** Now explicitly sets `auto_adjust=True`.

### Empty Tick Data

If tick data is empty, the market might be closed. Try:
- Use a longer period: `period="5d"`
- Use a different interval: `interval="5min"` instead of `1min`
- Check if market is open (US markets: 9:30 AM - 4:00 PM ET)

---

## Testing Infrastructure

### Quick Tests (`tests/quick_test_all.py`)

Runs minimal tests with 1 epoch training:

```bash
python tests/quick_test_all.py
```

**Tests:**
1. Module imports
2. Daily data loading
3. Tick data loading
4. Model creation
5. End-to-end training (1 epoch)
6. Error metrics calculation

**Runtime:** ~3-5 minutes

---

### Full Example Tests (`tests/test_examples.py`)

Runs all examples with full training:

```bash
python tests/test_examples.py
```

**Tests:**
- `timeseries_forecasting.py`
- `timeseries_detailed.py`
- `tick_data_forecasting.py`

**Runtime:** ~20-30 minutes

---

## Configuration

### Adjusting Training Parameters

All examples can be modified by editing these variables:

```python
# Data settings
TICKER = "AAPL"          # Stock ticker
PERIOD = "2y"            # Data period
SEQUENCE_LENGTH = 30     # Input sequence length
PREDICTION_HORIZON = 1   # Steps ahead to predict

# Training settings
EPOCHS = 30              # Number of training epochs
BATCH_SIZE = 32          # Batch size
LEARNING_RATE = 1e-4     # Learning rate
PATIENCE = 10            # Early stopping patience

# Model settings
MODEL_NAME = "gpt2"      # Base LLM
USE_LORA = True          # Use LoRA for efficiency
FREEZE_BACKBONE = False  # Freeze LLM backbone
```

---

## Tips for Best Results

### 1. More Data = Better Results

```python
# Instead of:
data = preprocessor.prepare_data(ticker="AAPL", start_date="2023-01-01")

# Try:
data = preprocessor.prepare_data(ticker="AAPL", start_date="2020-01-01")
```

### 2. Use Multiple Features

```python
# Instead of just returns:
features=["returns"]

# Try multiple indicators:
features=["returns", "sma_5", "sma_20", "volatility", "rsi"]
```

### 3. Tune Sequence Length

```python
# Shorter for high-frequency data:
sequence_length=10  # For 1-minute ticks

# Longer for daily data:
sequence_length=60  # For daily prices
```

### 4. Adjust Training Epochs

```python
# Quick test:
epochs=5

# Production:
epochs=50, patience=15
```

---

## Expected Output

### Successful Run Example

```
INFO:src.timeseries.financial_preprocessor:Loading data for AAPL
INFO:src.timeseries.financial_preprocessor:Loaded 502 data points

Data Statistics:
  Total sequences: 452
  Training sequences: 361
  Test sequences: 91
  Features: 5

Training model...
Epoch 1/30: 100%|████████| Loss: 0.0234
Epoch 2/30: 100%|████████| Loss: 0.0198
...

Test Metrics:
  MSE: 0.000123
  RMSE: 0.011090
  MAE: 0.008765
  Direction Accuracy: 54.32%

✅ All results saved to: outputs/timeseries_detailed/
```

---

## Next Steps

1. **Start with Yahoo Finance** (no API key needed)
2. **Run `timeseries_forecasting.py`** to verify setup
3. **Try `timeseries_detailed.py`** for comprehensive analysis
4. **Experiment with `tick_data_forecasting.py`** for intraday data
5. **Get API keys** for Alpha Vantage or Twelve Data for more data

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Run `python tests/quick_test_all.py` to isolate the problem
3. Check that all dependencies are installed: `pip install -r requirements.txt`
4. Verify your Python version: `python --version` (need 3.8+)
