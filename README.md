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
- **Financial Time Series**: Specialized module for stock market price prediction
- **Return Prediction**: Predict future returns based on historical prices/returns
- **LLM-based Forecasting**: Leverage language model architectures for sequential prediction
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

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (Parameter-Efficient Fine-Tuning)
- bitsandbytes (for quantization)
- pandas, numpy (for data processing)

## License

MIT License
