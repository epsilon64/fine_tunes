# Monte Carlo Strategy Performance Simulation

## Overview

This script performs Monte Carlo simulations to test strategy performance against a market index across different time horizons. It's designed to help understand the statistical properties of investment strategies and their likelihood of outperformance over various periods.

## Features

The simulation:
- **Simulates market index returns** using configurable mean and volatility parameters
- **Generates strategy returns** based on a target Information Ratio
- **Multi-dimensional analysis**: Tests across multiple Information Ratios (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0) and time periods (1, 3, 5, 10 years)
- **Outperformance matrix**: 2D matrix showing win rates across IR × Time Period dimensions
- **Computes three key performance metrics:**
  - **Information Ratio**: Measures risk-adjusted excess returns
  - **Relative Performance**: Simple difference between strategy and market returns
  - **Beta-Adjusted Relative Performance**: Jensen's Alpha accounting for market exposure
- **Measures outperformance frequency**: How often the strategy beats the market
- **Creates comprehensive visualizations**: Heatmaps, distribution plots, and comparison charts

## Usage

### Basic Usage

Simply run the script with default parameters to perform multi-dimensional analysis across 8 Information Ratios and 4 time periods:

```bash
python examples/monte_carlo_strategy_sim.py
```

This will run 32 combinations (8 IRs × 4 periods) with 10,000 simulations each (320,000 total simulations).

### Customizing Parameters

You can modify the simulation parameters by editing the `SimulationParams` in the `main()` function:

```python
params = SimulationParams(
    market_annual_return=0.08,      # 8% annual market return
    market_annual_vol=0.16,         # 16% annual market volatility
    strategy_info_ratio=0.5,        # Base IR (will be varied across info_ratios)
    strategy_tracking_error=0.05,   # 5% tracking error
    n_simulations=10000,            # Number of Monte Carlo runs per combination
    periods=[1, 3, 5, 10],          # Time periods to analyze (years)
    info_ratios=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]  # IRs to test
)
```

### Parameter Descriptions

#### Market Parameters
- `market_annual_return`: Expected annual return of the market index (e.g., 0.08 = 8%)
- `market_annual_vol`: Annual volatility of the market index (e.g., 0.16 = 16%)

#### Strategy Parameters
- `strategy_info_ratio`: Target Information Ratio for the strategy (measures skill level)
  - 0.5 = Good performance
  - 1.0 = Excellent performance
  - 0.0 = No skill (random)
- `strategy_tracking_error`: Annual tracking error of the strategy (e.g., 0.05 = 5%)
  - Lower values = more index-like behavior
  - Higher values = more active management

#### Simulation Parameters
- `n_simulations`: Number of Monte Carlo simulations to run per combination (default: 10,000)
- `periods`: List of time periods to analyze in years (default: [1, 3, 5, 10])
- `info_ratios`: List of Information Ratios to test (default: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])

## Output

### Console Output

The script outputs an **Outperformance Rate Matrix** showing the percentage of simulations where the strategy beats the market across all combinations of Information Ratio and time period:

```
OUTPERFORMANCE RATE MATRIX
(Percentage of simulations where strategy beats market)

  IR    1Y    3Y    5Y   10Y
0.05 50.6% 52.4% 51.4% 52.4%
0.10 53.0% 54.9% 56.7% 57.8%
0.20 56.1% 60.4% 64.2% 69.6%
0.30 60.7% 67.7% 71.0% 78.3%
0.40 64.6% 73.2% 78.6% 85.7%
0.50 67.5% 79.0% 84.2% 91.6%
0.75 75.2% 87.5% 93.1% 98.1%
1.00 82.6% 94.4% 97.8% 99.7%
```

This matrix clearly shows:
- **Horizontal trend** (across columns): Longer time periods increase outperformance probability
- **Vertical trend** (down rows): Higher Information Ratios lead to higher win rates
- **Key insight**: Even modest skill (IR 0.3-0.5) yields 70-91% win rates over 5-10 years

### Visualizations

A comprehensive visualization file `monte_carlo_multi_ir_results.png` is generated with six plots:

1. **Outperformance Rate Heatmap**: Color-coded matrix showing win rates across IR × Time Period
2. **Outperformance vs IR**: Line plots showing how win rates vary with Information Ratio for each period
3. **Outperformance vs Period**: Line plots showing how win rates increase over time for different IRs
4. **Realized IR Distributions**: Histograms showing distribution of realized IRs for different target IRs
5. **Mean Relative Performance**: Trends showing cumulative alpha growth over time

## Key Insights from Results

### Multi-Dimensional Analysis Findings

The outperformance rate matrix reveals critical patterns in strategy performance:

1. **Skill Matters**: Information Ratio is the dominant factor in long-term success
   - IR 0.05 (minimal skill): ~50-52% win rate across all periods (barely better than random)
   - IR 0.50 (moderate skill): 67.5% (1Y) → 91.6% (10Y)
   - IR 1.00 (high skill): 82.6% (1Y) → 99.7% (10Y)

2. **Time Diversification Effect**: Longer horizons dramatically increase success probability
   - For IR 0.50: +24.1 percentage points from 1Y to 10Y
   - For IR 0.20: +13.5 percentage points from 1Y to 10Y
   - Effect is stronger for higher IRs (more skill = more time benefit)

3. **Convergence to Certainty**: The combination of skill and time approaches certainty
   - IR 0.75 + 10Y = 98.1% win rate
   - IR 1.00 + 10Y = 99.7% win rate (virtually guaranteed outperformance)

4. **Patience Threshold**: Moderate skill requires patience to manifest
   - IR 0.30 needs ~5 years to reach 71% win rate
   - IR 0.50 needs ~3 years to reach 79% win rate
   - IR 0.75 needs just 1 year to reach 75% win rate

5. **Statistical Significance**: Longer periods make it easier to distinguish skill from luck
   - 1 year: Wide variation in outcomes even with skill
   - 10 years: Skill-based strategies consistently outperform

### Understanding Information Ratio

The **Information Ratio (IR)** is a key metric defined as:

```
IR = Alpha / Tracking Error
```

Where:
- **Alpha**: Excess return above the benchmark
- **Tracking Error**: Standard deviation of excess returns

An IR of 0.5 means the strategy is expected to generate 0.5 units of excess return per unit of risk taken.

### Understanding Beta-Adjusted Performance

**Beta-Adjusted Relative Performance** (Jensen's Alpha) accounts for the strategy's market exposure:

```
Alpha = Strategy Return - (Risk-free Rate + Beta × (Market Return - Risk-free Rate))
```

This metric isolates the manager's skill from simply taking market risk.

## Example Scenarios

### Conservative Strategy (IR = 0.3)

```python
params = SimulationParams(
    strategy_info_ratio=0.3,        # Lower skill level
    strategy_tracking_error=0.03,   # Lower tracking error (more conservative)
    n_simulations=10000,
    periods=[1, 3, 5, 10]
)
```

### Aggressive Strategy (IR = 1.0)

```python
params = SimulationParams(
    strategy_info_ratio=1.0,        # High skill level
    strategy_tracking_error=0.10,   # Higher tracking error (more active)
    n_simulations=10000,
    periods=[1, 3, 5, 10]
)
```

### Bear Market Scenario

```python
params = SimulationParams(
    market_annual_return=0.00,      # 0% market return
    market_annual_vol=0.25,         # 25% volatility (higher stress)
    strategy_info_ratio=0.5,
    strategy_tracking_error=0.05,
    n_simulations=10000,
    periods=[1, 3, 5, 10]
)
```

## Technical Details

### Simulation Methodology

1. **Market Returns**: Generated using geometric Brownian motion with specified mean and volatility

2. **Strategy Returns**: Modeled as:
   ```
   Strategy Return = Beta × Market Return + Alpha + Idiosyncratic Risk
   ```
   Where Alpha is calibrated to achieve the target Information Ratio

3. **Beta Estimation**: Beta is randomly sampled from N(1.0, 0.1) to reflect realistic market exposure with some variation

4. **Performance Metrics**: Calculated using standard financial formulas with annualization factors

### Statistical Properties

- **Random Seed**: Set to 42 for reproducibility
- **Daily Frequency**: All simulations use 252 trading days per year
- **Distributions**: Normal distributions for returns (reasonable for daily data)
- **Risk-Free Rate**: Assumed to be 0 for simplicity (can be modified if needed)

## Dependencies

- numpy
- pandas
- scipy
- matplotlib
- seaborn

All dependencies are included in the main `requirements.txt` file.

## Interpretation Guidelines

### When to Use This Simulation

1. **Strategy Evaluation**: Assess if a strategy's expected IR is sufficient for your goals
2. **Time Horizon Planning**: Understand how long you need to hold a strategy to see results
3. **Risk Assessment**: Evaluate the range of potential outcomes
4. **Patience Testing**: Determine the probability of underperformance in short periods

### Limitations

1. **Normal Distribution Assumption**: Real returns have fat tails and skewness
2. **Constant Parameters**: Assumes volatility and skill remain constant over time
3. **No Transaction Costs**: Does not account for trading costs, taxes, or fees
4. **Independence**: Assumes returns are independent (no autocorrelation)
5. **Simplified Beta**: Uses a simple beta model without industry or factor exposures

## Further Customization

You can extend the script by:

1. Adding more performance metrics (Sharpe ratio, Sortino ratio, max drawdown)
2. Implementing time-varying parameters (regime changes)
3. Adding transaction costs and fees
4. Using non-normal distributions (Student's t, mixture models)
5. Including multiple strategies for comparison
6. Adding Monte Carlo confidence intervals for the metrics

## References

- **Information Ratio**: Grinold, R. C., & Kahn, R. N. (1999). Active Portfolio Management
- **Jensen's Alpha**: Jensen, M. C. (1968). The Performance of Mutual Funds
- **Monte Carlo Methods**: Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering
