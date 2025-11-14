# Specialized Time Series Foundation Models

This guide covers using purpose-built time series foundation models like **Chronos** and **Lag-Llama**, which are pre-trained on diverse time series data and often outperform adapted general LLMs.

## 🎯 Why Use Specialized Models?

### General LLMs (GPT-2, Llama) for Time Series:
- ❌ Require fine-tuning on your data
- ❌ Not designed for numerical sequences
- ❌ Need custom projection layers
- ✅ Flexible for custom tasks
- ✅ Good with sufficient training data

### Specialized Models (Chronos, Lag-Llama):
- ✅ **Zero-shot ready** - work without training
- ✅ Pre-trained on diverse time series
- ✅ Purpose-built architecture
- ✅ Often better out-of-the-box
- ❌ Less flexible for custom tasks

---

## 📦 Available Models

### 1. Chronos (Amazon Research)

**Architecture:** T5-based encoder-decoder

**Key Features:**
- Tokenizes time series values
- Treats forecasting as language modeling
- Zero-shot forecasting
- Multiple model sizes

**Model Sizes:**

| Model | Parameters | Best For |
|-------|-----------|----------|
| `chronos-t5-tiny` | 8M | Quick experiments, CPU |
| `chronos-t5-mini` | 20M | Balanced speed/accuracy |
| `chronos-t5-small` | 46M | Good accuracy, fast |
| `chronos-t5-base` | 200M | High accuracy |
| `chronos-t5-large` | 710M | Best accuracy, needs GPU |

**Paper:** [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)

---

### 2. Lag-Llama

**Architecture:** Llama-based with temporal adaptations

**Key Features:**
- Probabilistic forecasting
- Uncertainty quantification
- Handles lagged features well
- ~1B parameters

**Best For:**
- Risk assessment
- Confidence intervals
- Probabilistic scenarios

**Paper:** [Lag-Llama: Towards Foundation Models for Time Series Forecasting](https://arxiv.org/abs/2310.08278)

---

## 🚀 Installation

### Chronos

```bash
# Install Chronos
pip install git+https://github.com/amazon-science/chronos-forecasting.git

# Models will auto-download on first use
```

### Lag-Llama

```bash
# Install GluonTS
pip install gluonts[torch]>=0.14.0

# Download model checkpoint from:
# https://github.com/time-series-foundation-models/lag-llama
```

---

## 💻 Usage

### Quick Start with Chronos

```python
from src.timeseries import ChronosModel, FinancialDataPreprocessor

# 1. Load model (first run downloads ~20-200MB)
model = ChronosModel(
    model_size="tiny",  # or "mini", "small", "base", "large"
    device="cpu"        # or "cuda" for GPU
)

# 2. Prepare your data
preprocessor = FinancialDataPreprocessor()
data = preprocessor.prepare_data(ticker="AAPL", start_date="2022-01-01")

# 3. Generate forecasts (no training needed!)
context = data['X_test'][0, :, 0]  # Historical data
forecast = model.predict(
    context=context,
    prediction_length=5,     # Predict 5 steps ahead
    num_samples=10,          # Generate 10 trajectories
    temperature=1.0          # Sampling temperature
)

# 4. Get median prediction
prediction = forecast.median(dim=0).values
```

### Using with TimeSeriesTrainer

```python
from src.timeseries import ChronosModel, TimeSeriesTrainer

# Load model
model = ChronosModel(model_size="small")

# Use with trainer (supports same interface)
trainer = TimeSeriesTrainer(model)

# Evaluate (no training needed, but you can fine-tune if desired)
metrics = trainer.evaluate(test_data=data)
```

### Uncertainty Quantification

```python
# Generate multiple samples for uncertainty
forecast = model.predict(
    context=context,
    prediction_length=5,
    num_samples=100,  # More samples = better uncertainty estimates
)

# Get statistics
median = forecast.median(dim=0).values
mean = forecast.mean(dim=0).values
std = forecast.std(dim=0).values
q10 = forecast.quantile(0.1, dim=0).values
q90 = forecast.quantile(0.9, dim=0).values

# Plot with confidence intervals
import matplotlib.pyplot as plt

plt.plot(mean, label='Mean Forecast')
plt.fill_between(
    range(len(mean)),
    q10, q90,
    alpha=0.3,
    label='80% Confidence'
)
plt.legend()
```

---

## 📊 Complete Example

```python
"""
Complete forecasting workflow with Chronos.
"""

from src.timeseries import (
    ChronosModel,
    FinancialDataPreprocessor,
    print_specialized_model_comparison,
)
import numpy as np

# Show available models
print_specialized_model_comparison()

# Prepare data
preprocessor = FinancialDataPreprocessor(
    sequence_length=30,
    prediction_horizon=5,
)

data = preprocessor.prepare_data(
    ticker="AAPL",
    start_date="2022-01-01",
    features=["returns"],
    train_ratio=0.8,
)

# Load Chronos model
model = ChronosModel(model_size="small")

# Generate forecasts
predictions = []
actuals = []

for i in range(len(data['X_test'])):
    context = data['X_test'][i, :, 0]

    # Zero-shot forecast
    forecast = model.predict(
        context=context,
        prediction_length=5,
        num_samples=20,
    )

    # Take median
    pred = forecast.median(dim=0).values.numpy()
    predictions.append(pred)

    actual = data['y_test'][i, :, 0]
    actuals.append(actual)

predictions = np.array(predictions)
actuals = np.array(actuals)

# Calculate metrics
mae = np.mean(np.abs(predictions - actuals))
direction_acc = np.mean(np.sign(predictions) == np.sign(actuals))

print(f"MAE: {mae:.6f}")
print(f"Direction Accuracy: {direction_acc:.2%}")
```

---

## 🎯 When to Use Which Model?

### Use Chronos When:
- ✅ You need quick results without training
- ✅ You have limited data
- ✅ You want a strong baseline
- ✅ Your data is similar to common time series patterns
- ✅ You need multiple horizon forecasts
- ✅ You want uncertainty quantification

### Use GPT-2/Custom Adaptation When:
- ✅ You have lots of training data (>10k samples)
- ✅ Your task is very specific/unusual
- ✅ You need custom features or multi-task learning
- ✅ You want to fine-tune on your exact distribution
- ✅ You're willing to invest in training time

### Use Lag-Llama When:
- ✅ You need probabilistic forecasts
- ✅ Uncertainty quantification is critical
- ✅ You have computational resources (1B params)
- ✅ You work with lagged features

---

## 🔬 Comparison: Chronos vs GPT-2 Adaptation

| Aspect | Chronos | GPT-2 Adaptation |
|--------|---------|------------------|
| **Training Required** | No (zero-shot) | Yes (fine-tuning) |
| **Setup Time** | < 5 minutes | Hours to days |
| **Data Requirements** | None (pre-trained) | Thousands of samples |
| **Architecture** | T5 (purpose-built) | GPT-2 (adapted) |
| **Performance** | Good out-of-box | Better with training |
| **Flexibility** | Limited | High |
| **Model Sizes** | 8M - 710M | 117M - 1.5B |
| **Best For** | Quick prototypes | Production systems |

---

## 📈 Performance Tips

### 1. Choose the Right Size

```python
# For CPU / Quick Experiments
model = ChronosModel(model_size="tiny")    # 8M params

# For Balanced Performance
model = ChronosModel(model_size="small")   # 46M params

# For Best Accuracy (GPU recommended)
model = ChronosModel(model_size="base")    # 200M params
```

### 2. Tune Sampling Parameters

```python
# For deterministic forecasts
forecast = model.predict(
    context=context,
    prediction_length=5,
    num_samples=1,
    temperature=0.0,
)

# For diverse scenarios
forecast = model.predict(
    context=context,
    prediction_length=5,
    num_samples=100,
    temperature=1.0,
)

# For conservative forecasts
forecast = model.predict(
    context=context,
    prediction_length=5,
    num_samples=50,
    temperature=0.5,
)
```

### 3. Handle Multiple Features

```python
# Chronos handles univariate series
# For multivariate, forecast each feature separately

for feature_idx in range(n_features):
    context = data[:, feature_idx]
    forecast = model.predict(context, prediction_length=5)
    predictions[:, feature_idx] = forecast.median(0).values
```

---

## 🐛 Troubleshooting

### Issue: "chronos not found"

**Solution:**
```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

### Issue: Model download fails

**Solution:**
- Check internet connection
- Ensure ~200MB free space for models
- Try smaller model first: `model_size="tiny"`

### Issue: Out of memory

**Solution:**
```python
# Use smaller model
model = ChronosModel(model_size="tiny")

# Reduce num_samples
forecast = model.predict(context, num_samples=10)  # Instead of 100

# Use CPU instead of GPU
model = ChronosModel(device="cpu")
```

### Issue: Slow predictions

**Solution:**
```python
# Use smaller model
model = ChronosModel(model_size="tiny")  # ~10x faster

# Reduce samples
forecast = model.predict(context, num_samples=1)

# Batch predictions (if using trainer)
predictions = trainer.predict(data, batch_size=32)
```

---

## 📚 Examples

Run the complete example:

```bash
python examples/specialized_models_forecasting.py
```

This example:
- Loads Chronos model
- Makes zero-shot forecasts on stock data
- Calculates comprehensive metrics
- Compares with baseline approaches
- Creates visualizations

---

## 🔗 Resources

### Chronos
- **GitHub:** https://github.com/amazon-science/chronos-forecasting
- **Paper:** https://arxiv.org/abs/2403.07815
- **HuggingFace:** https://huggingface.co/amazon/chronos-t5-tiny

### Lag-Llama
- **GitHub:** https://github.com/time-series-foundation-models/lag-llama
- **Paper:** https://arxiv.org/abs/2310.08278
- **HuggingFace:** https://huggingface.co/time-series-foundation-models

---

## 🎓 Next Steps

1. **Try the example:** `python examples/specialized_models_forecasting.py`
2. **Start with Chronos-tiny** for quick experiments
3. **Compare with fine-tuned GPT-2** (see `timeseries_forecasting.py`)
4. **Scale up to larger models** as needed
5. **Use in production** for consistent baseline performance

---

## 🤝 Contributing

Found a better specialized model? Want to add support for more architectures?

Contributions welcome! The framework is designed to easily add new specialized models.
