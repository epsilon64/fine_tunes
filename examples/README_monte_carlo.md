# Monte Carlo Strategy Performance Simulation

## Overview

This script performs Monte Carlo simulations to test strategy performance against a market index across different time horizons. It's designed to help understand the statistical properties of investment strategies and their likelihood of outperformance over various periods.

## Features

The simulation:
- **Simulates market index returns** using configurable mean and volatility parameters
- **Generates strategy returns** based on a target Information Ratio
- **Computes three key performance metrics:**
  - **Information Ratio**: Measures risk-adjusted excess returns
  - **Relative Performance**: Simple difference between strategy and market returns
  - **Beta-Adjusted Relative Performance**: Jensen's Alpha accounting for market exposure
- **Measures outperformance frequency**: How often the strategy beats the market
- **Analyzes multiple time horizons**: Default periods are 1, 3, 5, and 10 years
- **Creates comprehensive visualizations**: Distribution plots and comparison charts

## Usage

### Basic Usage

Simply run the script with default parameters:

```bash
python examples/monte_carlo_strategy_sim.py
```

### Customizing Parameters

You can modify the simulation parameters by editing the `SimulationParams` in the `main()` function:

```python
params = SimulationParams(
    market_annual_return=0.08,      # 8% annual market return
    market_annual_vol=0.16,         # 16% annual market volatility
    strategy_info_ratio=0.5,        # Information Ratio of 0.5
    strategy_tracking_error=0.05,   # 5% tracking error
    n_simulations=10000,            # Number of Monte Carlo runs
    periods=[1, 3, 5, 10]           # Time periods to analyze (years)
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
- `n_simulations`: Number of Monte Carlo simulations to run (default: 10,000)
- `periods`: List of time periods to analyze in years (default: [1, 3, 5, 10])

## Output

### Console Output

The script prints detailed statistics for each time period:

1. **Information Ratio Statistics**: Mean, median, standard deviation, and percentiles
2. **Relative Performance Statistics**: Distribution of excess returns vs. market
3. **Beta-Adjusted Performance**: Jensen's Alpha statistics
4. **Outperformance Analysis**: How often the strategy beats the market
5. **Average Returns**: Mean returns for both strategy and market

### Visualization

A comprehensive visualization file `monte_carlo_results.png` is generated with four plots:

1. **Information Ratio Distribution**: Shows the distribution of realized IRs across simulations
2. **Relative Performance Distribution**: Distribution of excess returns
3. **Outperformance Rate by Period**: Bar chart showing win rates for each time horizon
4. **Beta-Adjusted Performance Distribution**: Jensen's Alpha distribution

## Key Insights from Results

### Time Horizon Effects

The simulation demonstrates several important statistical properties:

1. **Convergence**: As the time horizon increases, the realized Information Ratio converges toward the expected value (standard deviation decreases)

2. **Outperformance Probability**: Longer time horizons increase the probability of outperformance
   - 1 year: ~67% outperformance rate
   - 10 years: ~92% outperformance rate

3. **Cumulative Alpha**: The cumulative benefit of skill compounds over time
   - Mean relative performance grows from ~2.7% (1Y) to ~64% (10Y)

4. **Statistical Significance**: Longer periods make it easier to distinguish skill from luck

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
