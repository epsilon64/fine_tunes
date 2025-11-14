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
data = preprocessor.load_stock_data("AAPL", start_date="2020-01-01")
train_data, test_data = preprocessor.prepare_sequences(data)

# Create and train model
ts_model = TimeSeriesLLM(model_name="gpt2", use_lora=True)
ts_model.train(train_data, epochs=10)

# Predict
predictions = ts_model.predict(test_data, steps=5)
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

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (Parameter-Efficient Fine-Tuning)
- bitsandbytes (for quantization)
- pandas, numpy (for data processing)

## License

MIT License
