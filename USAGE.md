# Usage Guide

This guide provides detailed instructions for using the LLM Fine-Tuning framework.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [General LLM Fine-Tuning](#general-llm-fine-tuning)
4. [Time Series Forecasting](#time-series-forecasting)
5. [Advanced Configuration](#advanced-configuration)
6. [LoRA Configuration](#lora-configuration)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd fine_tunes

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Start

### 1. Basic Fine-Tuning

The simplest way to fine-tune a model:

```python
from src.core.model_loader import ModelLoader
from src.core.trainer import FineTuner
from src.datasets.base_loader import prepare_dataset

# Load model
loader = ModelLoader()
model, tokenizer = loader.load_model("gpt2")

# Prepare dataset
dataset = prepare_dataset(
    dataset_name="wikitext",
    tokenizer=tokenizer,
    subset="wikitext-2-raw-v1"
)

# Train
trainer = FineTuner(model, tokenizer)
trainer.train(dataset)
```

### 2. LoRA Fine-Tuning

For efficient fine-tuning with LoRA:

```python
from src.core.lora import LoRAConfig, apply_lora

# Configure LoRA
lora_config = LoRAConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"]
)

# Apply to model
model = apply_lora(model, lora_config)

# Train as usual
trainer.train(dataset)
```

### 3. Time Series Forecasting

For financial time series prediction:

```python
from src.timeseries.financial_preprocessor import FinancialDataPreprocessor
from src.timeseries.ts_model import TimeSeriesLLM
from src.timeseries.ts_trainer import TimeSeriesTrainer

# Prepare data
preprocessor = FinancialDataPreprocessor()
data = preprocessor.prepare_data(ticker="AAPL")

# Create model
model = TimeSeriesLLM(model_name="gpt2")

# Train
trainer = TimeSeriesTrainer(model)
trainer.train(data)
```

## General LLM Fine-Tuning

### Loading Different Models

The framework supports various models:

```python
# GPT-2 variants
model, tokenizer = loader.load_model("gpt2")
model, tokenizer = loader.load_model("gpt2-medium")

# LLaMA models (requires authentication)
model, tokenizer = loader.load_model("meta-llama/Llama-2-7b-hf")

# Mistral models
model, tokenizer = loader.load_model("mistralai/Mistral-7B-v0.1")

# Phi models
model, tokenizer = loader.load_model("microsoft/phi-2")
```

### Quantization

Use quantization to reduce memory usage:

```python
# 4-bit quantization (recommended)
model, tokenizer = loader.load_model(
    "gpt2",
    quantization="4bit"
)

# 8-bit quantization
model, tokenizer = loader.load_model(
    "gpt2",
    quantization="8bit"
)
```

Note: Quantization requires `bitsandbytes` and CUDA.

### Custom Datasets

Load your own text data:

```python
from src.datasets.base_loader import TextDatasetLoader

loader = TextDatasetLoader(tokenizer)

# From file
dataset = loader.load_from_text_file(
    "my_data.txt",
    delimiter="\n\n"
)

# Prepare for training
dataset = loader.prepare_dataset(dataset)
```

## Time Series Forecasting

### Loading Financial Data

```python
from src.timeseries.financial_preprocessor import FinancialDataPreprocessor

preprocessor = FinancialDataPreprocessor(
    sequence_length=30,
    prediction_horizon=1
)

# Single stock
data = preprocessor.load_stock_data(
    ticker="AAPL",
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# With technical indicators
data = preprocessor.calculate_technical_features(data)
```

### Using Multiple Features

```python
data = preprocessor.prepare_data(
    ticker="AAPL",
    features=[
        "returns",
        "sma_5",
        "sma_20",
        "volatility",
        "rsi"
    ]
)
```

### Model Types

Choose between basic and adaptive models:

```python
from src.timeseries.ts_model import TimeSeriesLLM, AdaptiveTimeSeriesLLM

# Basic model
model = TimeSeriesLLM(
    model_name="gpt2",
    d_input=5,  # Number of features
    d_output=1
)

# Adaptive model with temporal encoding (recommended)
model = AdaptiveTimeSeriesLLM(
    model_name="gpt2",
    d_input=5,
    d_output=1,
    use_temporal_encoding=True
)
```

### Making Predictions

```python
# Train the model
trainer = TimeSeriesTrainer(model)
trainer.train(data)

# Evaluate
metrics = trainer.evaluate(data)

# Predict future values
predictions = trainer.predict(
    data["X_test"],
    steps_ahead=5
)
```

## Advanced Configuration

### Using Config Files

Load configuration from YAML:

```python
import yaml

with open("config/default_config.yaml") as f:
    config = yaml.safe_load(f)

# Use config values
model, tokenizer = loader.load_model(
    config["model"]["name"],
    quantization=config["model"]["quantization"]
)
```

### Training Arguments

Customize training parameters:

```python
trainer.train(
    train_dataset=train_data,
    eval_dataset=eval_data,
    num_epochs=5,
    batch_size=8,
    learning_rate=2e-4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    save_steps=500,
    logging_steps=10,
    fp16=True
)
```

### Early Stopping

For time series models:

```python
trainer.train(
    train_data=data,
    val_data=data,
    patience=10,  # Stop if no improvement for 10 epochs
    save_best=True  # Save best model
)
```

## LoRA Configuration

### Selecting Target Modules

Choose which parts of the model to fine-tune:

```python
from src.core.lora import LoRAModuleSelector, create_lora_config_for_model

# Only attention layers
config = create_lora_config_for_model(
    model_type="gpt2",
    include_attention=True,
    include_mlp=False
)

# Both attention and MLP
config = create_lora_config_for_model(
    model_type="gpt2",
    include_attention=True,
    include_mlp=True
)

# Custom modules
config = LoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj", "c_fc"]
)
```

### LoRA Parameters

Key parameters to adjust:

- **r** (rank): Controls adapter capacity. Common values: 4, 8, 16, 32
  - Lower r = fewer parameters, faster training
  - Higher r = more capacity, potentially better performance

- **lora_alpha**: Scaling factor. Typically 2-4x the rank
  - Higher alpha = stronger adaptation

- **lora_dropout**: Dropout for LoRA layers (0.0-0.2)

Example configurations:

```python
# Minimal (fastest, least memory)
LoRAConfig(r=4, lora_alpha=16)

# Balanced (recommended)
LoRAConfig(r=8, lora_alpha=32)

# Maximum capacity
LoRAConfig(r=32, lora_alpha=64)
```

## Running Examples

```bash
# Basic fine-tuning
python examples/basic_finetuning.py

# LoRA fine-tuning
python examples/lora_finetuning.py

# Time series forecasting
python examples/timeseries_forecasting.py
```

## Troubleshooting

### CUDA Out of Memory

- Use quantization: `quantization="4bit"`
- Reduce batch size
- Enable gradient checkpointing (enabled by default)
- Use gradient accumulation

### Slow Training

- Increase batch size
- Reduce sequence length
- Use FP16 training (enabled by default)
- Use LoRA instead of full fine-tuning

### Poor Time Series Performance

- Increase sequence length
- Add more features (technical indicators)
- Try adaptive model with temporal encoding
- Increase training epochs
- Adjust learning rate

## Next Steps

- Check `examples/` for complete working examples
- Review `config/` for configuration templates
- See notebooks for interactive demos
- Read the code documentation for advanced usage
