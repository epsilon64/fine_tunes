# Time Series Forecast Visualization Guide

This guide covers the comprehensive visualization utilities for analyzing time series forecasts and comparing predictions against realized values.

## Overview

The `ForecastVisualizer` class provides professional-quality visualizations for time series forecast analysis, including:

1. **Price Reconstruction**: Convert return predictions back to price levels
2. **Comprehensive Comparison**: 4-panel analysis with multiple metrics
3. **Multi-Horizon Analysis**: Track accuracy across different forecast horizons
4. **Predictions vs Actuals**: Time series overlay with embedded metrics
5. **Multi-Step Forecasts**: Visualize multiple-step-ahead predictions

All visualizations are saved as high-quality 300 DPI PNG files with professional styling.

---

## Installation

The visualization module is included in the main package:

```python
from src.timeseries import ForecastVisualizer, plot_multi_step_forecast
```

**Requirements:**
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- numpy >= 1.24.0
- pandas >= 2.0.0

---

## Quick Start

```python
from src.timeseries import ForecastVisualizer
import numpy as np

# Your forecast data
predictions = np.array([...])  # Shape: (n_samples, n_features)
actuals = np.array([...])      # Shape: (n_samples, n_features)

# Create visualizer
visualizer = ForecastVisualizer(output_dir="./outputs")

# Generate comprehensive analysis
visualizer.plot_comprehensive_comparison(
    predictions=predictions,
    actuals=actuals,
    title="My Forecast Analysis"
)
```

---

## Key Visualization Methods

### 1. Price Reconstruction from Returns

**Purpose**: Convert return predictions back to price levels and compare with realized prices.

**Use Case**: When your model predicts returns but you want to visualize actual vs predicted price paths.

```python
visualizer.plot_price_reconstruction(
    returns_predictions=returns_pred,  # Shape: (n_samples, n_features)
    returns_actuals=returns_actual,    # Shape: (n_samples, n_features)
    initial_price=100.0,               # Starting price level
    dates=date_index,                  # Optional: datetime index
    title="AAPL - Price Forecast vs Realized Prices"
)
```

**Output:**
- Top panel: Predicted vs actual price paths
- Bottom panel: Cumulative returns comparison
- Embedded metrics: MAE, RMSE, MAPE (in dollars)

**Example:**
```
Initial price: $100
Returns predictions: [0.02, -0.01, 0.03, ...]
→ Price path: [$102.00, $100.98, $104.01, ...]
```

---

### 2. Comprehensive Comparison (4-Panel Analysis)

**Purpose**: Complete forecast evaluation in a single view.

```python
visualizer.plot_comprehensive_comparison(
    predictions=predictions,
    actuals=actuals,
    dates=date_index,  # Optional
    title="Comprehensive Forecast Analysis"
)
```

**Panels:**
1. **Time Series Overlay**: Predicted vs actual values over time
2. **Scatter Plot**: Correlation with R² score
3. **Error Distribution**: Histogram of prediction errors
4. **Cumulative Absolute Error**: Error accumulation over time

**Metrics Displayed:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Direction Accuracy (for trading strategies)

---

### 3. Multi-Horizon Accuracy Analysis

**Purpose**: Compare forecast accuracy across different time horizons.

**Use Case**: Multi-step forecasting where you want to see how accuracy degrades with forecast distance.

```python
predictions = np.array([...])  # Shape: (n_samples, n_horizons, n_features)
actuals = np.array([...])      # Shape: (n_samples, n_horizons, n_features)

visualizer.plot_forecast_horizon_analysis(
    predictions=predictions,
    actuals=actuals,
    horizon_names=["H+1", "H+2", "H+3", "H+5"],
    title="Multi-Horizon Forecast Analysis"
)
```

**Output (3 panels):**
1. MAE by horizon
2. RMSE by horizon
3. Direction Accuracy by horizon

**Typical Pattern:**
- H+1: MAE = 0.012, Direction Acc = 58%
- H+2: MAE = 0.018, Direction Acc = 55%
- H+3: MAE = 0.025, Direction Acc = 52%
(Accuracy typically degrades with forecast distance)

---

### 4. Predictions vs Actuals (with Confidence Intervals)

**Purpose**: Basic time series comparison with optional uncertainty bands.

```python
visualizer.plot_predictions_vs_actual(
    predictions=predictions,
    actuals=actuals,
    dates=date_index,
    title="1-Day Ahead Return Forecasts",
    show_confidence=True,
    confidence_lower=lower_bound,  # 10th percentile
    confidence_upper=upper_bound,  # 90th percentile
)
```

**Features:**
- Time series overlay
- Confidence interval shading
- Embedded metrics (MAE, RMSE, Direction Accuracy)
- Automatic date formatting

---

### 5. Multi-Step Forecast Visualization

**Purpose**: Show multiple forecast horizons in a single plot.

```python
from src.timeseries import plot_multi_step_forecast

plot_multi_step_forecast(
    historical_data=context,           # Shape: (seq_length,)
    predictions=forecast,              # Shape: (n_horizons,)
    actuals=actual_future,             # Shape: (n_horizons,) or None
    horizon_names=["T+1", "T+2", "T+3", "T+5"],
    title="5-Day Ahead Forecasts",
    save_path="./outputs/multi_step_forecast.png"
)
```

**Output:**
- Historical context (gray line)
- Forecast markers (blue circles)
- Actual values if provided (orange line)
- Vertical line separating history from forecast

---

## Integration with Examples

All examples now include visualization automatically:

### timeseries_detailed.py

```python
# ... training code ...

# Visualizations automatically generated:
# 1. Error analysis plots (existing)
# 2. Comprehensive comparison (NEW)
# 3. Price reconstruction from returns (NEW)
```

**New Outputs:**
- `comprehensive_comparison.png`: 4-panel analysis
- `price_reconstruction.png`: Predicted vs realized prices

---

### tick_data_forecasting.py

```python
# ... training code ...

# Visualizations automatically generated:
# 1. Training history
# 2. Tick forecasting results
# 3. Comprehensive comparison (NEW)
# 4. Price reconstruction from tick returns (NEW)
```

**New Outputs:**
- `comprehensive_comparison.png`: Tick forecast analysis
- `price_reconstruction.png`: Intraday price paths

---

### specialized_models_forecasting.py (Chronos)

```python
# ... Chronos model code ...

# Visualizations automatically generated:
# 1. Basic forecasts plot
# 2. Comprehensive comparison (NEW)
# 3. Multi-horizon analysis (NEW) - if prediction_horizon > 1
# 4. Price reconstruction (NEW)
```

**New Outputs:**
- `comprehensive_comparison.png`: Zero-shot forecast analysis
- `forecast_horizon_analysis.png`: Multi-step accuracy
- `price_reconstruction.png`: Zero-shot price predictions

---

## Customization

### Output Directory

```python
visualizer = ForecastVisualizer(output_dir="./my_results")
```

### Plot Styling

```python
# Change figure size
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 12)

# Use different style
import seaborn as sns
sns.set_style("darkgrid")
```

### Custom Metrics

```python
# Calculate your own metrics
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

# Add to plot title
title = f"My Forecast (MSE={mse:.4f}, R²={r2:.4f})"
```

---

## Advanced Usage

### Batch Processing Multiple Tickers

```python
visualizer = ForecastVisualizer(output_dir="./outputs")

tickers = ["AAPL", "GOOGL", "MSFT"]
for ticker in tickers:
    # ... get predictions and actuals for ticker ...

    visualizer.plot_price_reconstruction(
        returns_predictions=predictions,
        returns_actuals=actuals,
        initial_price=initial_prices[ticker],
        title=f"{ticker} - Forecast vs Realized"
    )
    # Saves as: ./outputs/price_reconstruction_{ticker}.png (auto-numbered)
```

### Comparing Multiple Models

```python
models = ["GPT-2", "Chronos-Small", "Lag-Llama"]
predictions_dict = {...}  # predictions for each model

fig, axes = plt.subplots(len(models), 1, figsize=(12, 4*len(models)))

for i, model_name in enumerate(models):
    preds = predictions_dict[model_name]
    axes[i].plot(actuals, label='Actual', alpha=0.7)
    axes[i].plot(preds, label=f'{model_name} Predicted', alpha=0.7)
    axes[i].set_title(f'{model_name} Forecasts')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./outputs/model_comparison.png", dpi=300)
```

---

## Metrics Reference

All metrics are automatically calculated and displayed:

| Metric | Description | Typical Range | Good Value |
|--------|-------------|---------------|------------|
| **MAE** | Mean Absolute Error | 0 to ∞ | Lower is better |
| **RMSE** | Root Mean Squared Error | 0 to ∞ | Lower is better |
| **MAPE** | Mean Absolute Percentage Error | 0% to ∞ | < 10% is good |
| **R²** | Coefficient of Determination | -∞ to 1 | > 0.5 is decent |
| **Direction Accuracy** | % correct directional predictions | 0% to 100% | > 55% is good |

**For Financial Returns:**
- MAE < 0.02 (2%): Very good
- MAE 0.02-0.05 (2-5%): Good
- MAE > 0.05 (>5%): Needs improvement

**For Direction Accuracy:**
- > 55%: Potentially profitable
- 50%: Random (coin flip)
- < 45%: Inverse signals might be useful

---

## Troubleshooting

### Issue: Plots look squashed

**Solution:**
```python
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
```

### Issue: Memory error with large datasets

**Solution:**
```python
# Downsample for visualization
n_plot = 1000
stride = len(predictions) // n_plot
visualizer.plot_comprehensive_comparison(
    predictions=predictions[::stride],
    actuals=actuals[::stride]
)
```

### Issue: Dates not displaying correctly

**Solution:**
```python
import pandas as pd
dates = pd.to_datetime(dates)  # Ensure datetime type

# Rotate date labels
import matplotlib.pyplot as plt
plt.xticks(rotation=45)
```

---

## Examples in Action

### Example 1: Daily Stock Returns

```python
from src.timeseries import (
    FinancialDataPreprocessor,
    AdaptiveTimeSeriesLLM,
    TimeSeriesTrainer,
    ForecastVisualizer
)

# Prepare data
preprocessor = FinancialDataPreprocessor(sequence_length=30, prediction_horizon=1)
data = preprocessor.prepare_data(ticker="AAPL", start_date="2022-01-01")

# Train model
model = AdaptiveTimeSeriesLLM(...)
trainer = TimeSeriesTrainer(model)
trainer.train(data, epochs=30)

# Predict
predictions = trainer.predict(data["X_test"])
actuals = data["y_test"][:, 0, 0]
predictions = predictions[:, 0, 0]

# Visualize
visualizer = ForecastVisualizer(output_dir="./outputs")

# 1. Comprehensive analysis
visualizer.plot_comprehensive_comparison(
    predictions=predictions.reshape(-1, 1),
    actuals=actuals.reshape(-1, 1),
    title="AAPL Daily Return Forecasts"
)

# 2. Price reconstruction
initial_price = 150.0  # AAPL price at test start
visualizer.plot_price_reconstruction(
    returns_predictions=predictions.reshape(-1, 1),
    returns_actuals=actuals.reshape(-1, 1),
    initial_price=initial_price,
    title="AAPL - Predicted vs Realized Prices"
)
```

**Output Files:**
- `./outputs/comprehensive_comparison.png`
- `./outputs/price_reconstruction.png`

---

### Example 2: Multi-Step Forecasting

```python
# 5-day ahead forecasting
preprocessor = FinancialDataPreprocessor(sequence_length=30, prediction_horizon=5)
data = preprocessor.prepare_data(ticker="AAPL")

# ... train model ...

# Predictions shape: (n_samples, 5, 1)
predictions = trainer.predict(data["X_test"], steps_ahead=5)
actuals = data["y_test"]  # Shape: (n_samples, 5, 1)

# Visualize
visualizer = ForecastVisualizer()

# Multi-horizon analysis
visualizer.plot_forecast_horizon_analysis(
    predictions=predictions,
    actuals=actuals,
    horizon_names=["Day+1", "Day+2", "Day+3", "Day+4", "Day+5"]
)

# Shows how accuracy degrades over forecast horizon
```

---

### Example 3: Chronos Zero-Shot Forecasting

```python
from src.timeseries import ChronosModel, ForecastVisualizer

model = ChronosModel(model_size="small")

predictions = []
for i in range(len(test_data)):
    forecast = model.predict(
        context=test_data[i],
        prediction_length=5,
        num_samples=20
    )
    predictions.append(forecast.median(dim=0).values)

predictions = np.array(predictions)  # Shape: (n_samples, 5)
actuals = actual_returns  # Shape: (n_samples, 5)

# Visualize
visualizer = ForecastVisualizer()
visualizer.plot_forecast_horizon_analysis(
    predictions=predictions,
    actuals=actuals,
    title="Chronos Zero-Shot Multi-Step Forecasts"
)
```

---

## Best Practices

1. **Always reshape single-feature predictions**:
   ```python
   predictions.reshape(-1, 1)  # From (n,) to (n, 1)
   ```

2. **Use price reconstruction for trading strategies**:
   - Returns are hard to interpret
   - Prices show actual P&L

3. **Multi-horizon analysis for production**:
   - Understand accuracy degradation
   - Set appropriate forecast horizons

4. **Save all plots**:
   - Include in reports
   - Track improvement over time

5. **Include dates when available**:
   - Easier to identify patterns
   - Better context for results

---

## Next Steps

1. **Run examples**: All examples now include these visualizations
   ```bash
   python examples/timeseries_detailed.py
   python examples/specialized_models_forecasting.py
   ```

2. **Customize for your use case**: Modify plot titles, metrics, styling

3. **Integrate into production**: Use for model monitoring and reporting

4. **Extend functionality**: Add custom visualizations for your specific needs

---

## Contributing

Want to add new visualization methods? The `ForecastVisualizer` class is designed to be extensible:

```python
class ForecastVisualizer:
    def plot_your_custom_viz(self, ...):
        """Your custom visualization."""
        # Your implementation
        plt.savefig(f"{self.output_dir}/your_viz.png", dpi=300)
```

---

## References

- **Matplotlib Documentation**: https://matplotlib.org/
- **Seaborn Gallery**: https://seaborn.pydata.org/examples/index.html
- **Time Series Metrics**: See `TESTING.md` for detailed metric explanations

---

**Happy Forecasting! 📈**
