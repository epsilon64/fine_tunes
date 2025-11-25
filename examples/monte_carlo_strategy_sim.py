"""
Monte Carlo Simulation for Strategy Performance Analysis

This script performs Monte Carlo simulations to test strategy performance against
a market index across different time horizons. It computes key performance metrics
including information ratio, relative performance, and beta-adjusted relative performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


@dataclass
class SimulationParams:
    """Parameters for Monte Carlo simulation"""
    # Market parameters
    market_annual_return: float = 0.08  # 8% annual return
    market_annual_vol: float = 0.16     # 16% annual volatility

    # Strategy parameters
    strategy_info_ratio: float = 0.5    # Expected information ratio
    strategy_tracking_error: float = 0.05  # 5% tracking error (annual)

    # Simulation parameters
    n_simulations: int = 10000
    trading_days_per_year: int = 252

    # Time periods to analyze (in years)
    periods: List[int] = None

    def __post_init__(self):
        if self.periods is None:
            self.periods = [1, 3, 5, 10]


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    information_ratio: float
    relative_performance: float  # Strategy return - benchmark return
    beta_adjusted_relative_performance: float
    strategy_return: float
    market_return: float
    beta: float
    strategy_vol: float
    market_vol: float


class MonteCarloSimulator:
    """Monte Carlo simulator for strategy performance analysis"""

    def __init__(self, params: SimulationParams):
        self.params = params
        np.random.seed(42)  # For reproducibility

    def simulate_market_returns(self, n_days: int) -> np.ndarray:
        """
        Simulate daily market returns using geometric Brownian motion

        Args:
            n_days: Number of trading days to simulate

        Returns:
            Array of daily returns
        """
        # Convert annual parameters to daily
        daily_return = self.params.market_annual_return / self.params.trading_days_per_year
        daily_vol = self.params.market_annual_vol / np.sqrt(self.params.trading_days_per_year)

        # Generate random returns
        returns = np.random.normal(daily_return, daily_vol, n_days)

        return returns

    def simulate_strategy_returns(self, market_returns: np.ndarray) -> np.ndarray:
        """
        Simulate strategy returns based on market returns and target information ratio

        The strategy has:
        1. A market component (beta exposure)
        2. An alpha component (skill-based returns)

        Args:
            market_returns: Array of market returns

        Returns:
            Array of strategy returns
        """
        n_days = len(market_returns)

        # Strategy should have some market exposure (assume beta ~1 with some variation)
        beta = np.random.normal(1.0, 0.1)

        # Convert annual tracking error to daily
        daily_tracking_error = self.params.strategy_tracking_error / np.sqrt(self.params.trading_days_per_year)

        # Expected alpha from information ratio: IR = alpha / tracking_error
        daily_alpha = (self.params.strategy_info_ratio * self.params.strategy_tracking_error) / self.params.trading_days_per_year

        # Generate idiosyncratic returns (strategy-specific)
        idiosyncratic_returns = np.random.normal(daily_alpha, daily_tracking_error, n_days)

        # Strategy returns = beta * market returns + alpha + idiosyncratic risk
        strategy_returns = beta * market_returns + idiosyncratic_returns

        return strategy_returns, beta

    def calculate_metrics(self, strategy_returns: np.ndarray,
                         market_returns: np.ndarray,
                         true_beta: float) -> PerformanceMetrics:
        """
        Calculate performance metrics for a single simulation

        Args:
            strategy_returns: Array of strategy returns
            market_returns: Array of market returns
            true_beta: True beta used in simulation

        Returns:
            PerformanceMetrics object
        """
        # Calculate cumulative returns
        strategy_total_return = np.prod(1 + strategy_returns) - 1
        market_total_return = np.prod(1 + market_returns) - 1

        # Calculate volatilities (annualized)
        strategy_vol = np.std(strategy_returns) * np.sqrt(self.params.trading_days_per_year)
        market_vol = np.std(market_returns) * np.sqrt(self.params.trading_days_per_year)

        # Calculate excess returns (strategy vs market)
        excess_returns = strategy_returns - market_returns

        # Information Ratio
        if np.std(excess_returns) > 1e-10:
            information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.params.trading_days_per_year)
        else:
            information_ratio = 0.0

        # Relative Performance (simple difference)
        relative_performance = strategy_total_return - market_total_return

        # Estimate beta from returns (may differ from true beta due to randomness)
        if np.var(market_returns) > 1e-10:
            estimated_beta = np.cov(strategy_returns, market_returns)[0, 1] / np.var(market_returns)
        else:
            estimated_beta = true_beta

        # Beta-adjusted relative performance (Jensen's alpha)
        # Alpha = Strategy Return - (Risk-free rate + Beta * (Market Return - Risk-free rate))
        # Assuming risk-free rate = 0 for simplicity
        beta_adjusted_relative_performance = strategy_total_return - estimated_beta * market_total_return

        return PerformanceMetrics(
            information_ratio=information_ratio,
            relative_performance=relative_performance,
            beta_adjusted_relative_performance=beta_adjusted_relative_performance,
            strategy_return=strategy_total_return,
            market_return=market_total_return,
            beta=estimated_beta,
            strategy_vol=strategy_vol,
            market_vol=market_vol
        )

    def run_single_simulation(self, n_days: int) -> PerformanceMetrics:
        """Run a single Monte Carlo simulation"""
        # Simulate market returns
        market_returns = self.simulate_market_returns(n_days)

        # Simulate strategy returns
        strategy_returns, true_beta = self.simulate_strategy_returns(market_returns)

        # Calculate metrics
        metrics = self.calculate_metrics(strategy_returns, market_returns, true_beta)

        return metrics

    def run_simulations(self, period_years: int) -> List[PerformanceMetrics]:
        """
        Run multiple Monte Carlo simulations for a given time period

        Args:
            period_years: Time period in years

        Returns:
            List of PerformanceMetrics for each simulation
        """
        n_days = period_years * self.params.trading_days_per_year
        results = []

        print(f"\nRunning {self.params.n_simulations} simulations for {period_years} year(s)...")

        for i in range(self.params.n_simulations):
            if (i + 1) % 1000 == 0:
                print(f"  Completed {i + 1}/{self.params.n_simulations} simulations")

            metrics = self.run_single_simulation(n_days)
            results.append(metrics)

        return results

    def analyze_results(self, results: List[PerformanceMetrics], period_years: int) -> Dict:
        """
        Analyze simulation results and compute summary statistics

        Args:
            results: List of PerformanceMetrics from simulations
            period_years: Time period in years

        Returns:
            Dictionary of summary statistics
        """
        # Extract metrics into arrays
        information_ratios = np.array([r.information_ratio for r in results])
        relative_performances = np.array([r.relative_performance for r in results])
        beta_adj_performances = np.array([r.beta_adjusted_relative_performance for r in results])
        strategy_returns = np.array([r.strategy_return for r in results])
        market_returns = np.array([r.market_return for r in results])

        # Count outperformance
        outperformance_count = np.sum(strategy_returns > market_returns)
        outperformance_rate = outperformance_count / len(results)

        # Calculate statistics for each metric
        summary = {
            'period_years': period_years,
            'n_simulations': len(results),

            # Information Ratio statistics
            'ir_mean': np.mean(information_ratios),
            'ir_median': np.median(information_ratios),
            'ir_std': np.std(information_ratios),
            'ir_percentile_5': np.percentile(information_ratios, 5),
            'ir_percentile_95': np.percentile(information_ratios, 95),

            # Relative Performance statistics (in %)
            'rel_perf_mean': np.mean(relative_performances) * 100,
            'rel_perf_median': np.median(relative_performances) * 100,
            'rel_perf_std': np.std(relative_performances) * 100,
            'rel_perf_percentile_5': np.percentile(relative_performances, 5) * 100,
            'rel_perf_percentile_95': np.percentile(relative_performances, 95) * 100,

            # Beta-adjusted relative performance statistics (in %)
            'beta_adj_mean': np.mean(beta_adj_performances) * 100,
            'beta_adj_median': np.median(beta_adj_performances) * 100,
            'beta_adj_std': np.std(beta_adj_performances) * 100,
            'beta_adj_percentile_5': np.percentile(beta_adj_performances, 5) * 100,
            'beta_adj_percentile_95': np.percentile(beta_adj_performances, 95) * 100,

            # Outperformance statistics
            'outperformance_count': outperformance_count,
            'outperformance_rate': outperformance_rate * 100,  # in %

            # Average returns (in %)
            'avg_strategy_return': np.mean(strategy_returns) * 100,
            'avg_market_return': np.mean(market_returns) * 100,
        }

        return summary

    def print_summary(self, summary: Dict):
        """Print summary statistics in a readable format"""
        print(f"\n{'='*70}")
        print(f"RESULTS FOR {summary['period_years']} YEAR(S) ({summary['n_simulations']} simulations)")
        print(f"{'='*70}")

        print(f"\n{'-'*70}")
        print("INFORMATION RATIO")
        print(f"{'-'*70}")
        print(f"  Mean:               {summary['ir_mean']:8.3f}")
        print(f"  Median:             {summary['ir_median']:8.3f}")
        print(f"  Std Dev:            {summary['ir_std']:8.3f}")
        print(f"  5th Percentile:     {summary['ir_percentile_5']:8.3f}")
        print(f"  95th Percentile:    {summary['ir_percentile_95']:8.3f}")

        print(f"\n{'-'*70}")
        print("RELATIVE PERFORMANCE (Strategy Return - Market Return)")
        print(f"{'-'*70}")
        print(f"  Mean:               {summary['rel_perf_mean']:8.2f}%")
        print(f"  Median:             {summary['rel_perf_median']:8.2f}%")
        print(f"  Std Dev:            {summary['rel_perf_std']:8.2f}%")
        print(f"  5th Percentile:     {summary['rel_perf_percentile_5']:8.2f}%")
        print(f"  95th Percentile:    {summary['rel_perf_percentile_95']:8.2f}%")

        print(f"\n{'-'*70}")
        print("BETA-ADJUSTED RELATIVE PERFORMANCE (Jensen's Alpha)")
        print(f"{'-'*70}")
        print(f"  Mean:               {summary['beta_adj_mean']:8.2f}%")
        print(f"  Median:             {summary['beta_adj_median']:8.2f}%")
        print(f"  Std Dev:            {summary['beta_adj_std']:8.2f}%")
        print(f"  5th Percentile:     {summary['beta_adj_percentile_5']:8.2f}%")
        print(f"  95th Percentile:    {summary['beta_adj_percentile_95']:8.2f}%")

        print(f"\n{'-'*70}")
        print("OUTPERFORMANCE ANALYSIS")
        print(f"{'-'*70}")
        print(f"  Times Strategy Beat Market:    {summary['outperformance_count']:6d}")
        print(f"  Outperformance Rate:           {summary['outperformance_rate']:7.2f}%")

        print(f"\n{'-'*70}")
        print("AVERAGE RETURNS")
        print(f"{'-'*70}")
        print(f"  Strategy:           {summary['avg_strategy_return']:8.2f}%")
        print(f"  Market:             {summary['avg_market_return']:8.2f}%")

    def create_visualizations(self, all_results: Dict[int, List[PerformanceMetrics]],
                            save_path: str = 'monte_carlo_results.png'):
        """
        Create visualization plots for simulation results

        Args:
            all_results: Dictionary mapping period_years to list of PerformanceMetrics
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Monte Carlo Simulation Results Across Time Horizons',
                     fontsize=16, fontweight='bold')

        periods = sorted(all_results.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(periods)))

        # Plot 1: Information Ratio Distribution
        ax = axes[0, 0]
        for period, color in zip(periods, colors):
            results = all_results[period]
            irs = [r.information_ratio for r in results]
            ax.hist(irs, bins=50, alpha=0.5, label=f'{period}Y', color=color, density=True)
        ax.axvline(self.params.strategy_info_ratio, color='red', linestyle='--',
                   linewidth=2, label='Expected IR')
        ax.set_xlabel('Information Ratio', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Information Ratio Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Relative Performance Distribution
        ax = axes[0, 1]
        for period, color in zip(periods, colors):
            results = all_results[period]
            rel_perfs = [r.relative_performance * 100 for r in results]
            ax.hist(rel_perfs, bins=50, alpha=0.5, label=f'{period}Y', color=color, density=True)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax.set_xlabel('Relative Performance (%)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Relative Performance Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Outperformance Rate by Period
        ax = axes[1, 0]
        outperf_rates = []
        for period in periods:
            results = all_results[period]
            strategy_rets = np.array([r.strategy_return for r in results])
            market_rets = np.array([r.market_return for r in results])
            outperf_rate = np.sum(strategy_rets > market_rets) / len(results) * 100
            outperf_rates.append(outperf_rate)

        bars = ax.bar([f'{p}Y' for p in periods], outperf_rates, color=colors, alpha=0.7)
        ax.axhline(50, color='red', linestyle='--', linewidth=2, label='50% (Random)')
        ax.set_ylabel('Outperformance Rate (%)', fontsize=12)
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_title('Strategy Outperformance Rate by Period', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        for bar, rate in zip(bars, outperf_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Plot 4: Beta-Adjusted Performance
        ax = axes[1, 1]
        for period, color in zip(periods, colors):
            results = all_results[period]
            beta_adj = [r.beta_adjusted_relative_performance * 100 for r in results]
            ax.hist(beta_adj, bins=50, alpha=0.5, label=f'{period}Y', color=color, density=True)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Alpha')
        ax.set_xlabel('Beta-Adjusted Relative Performance (%)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Beta-Adjusted Performance Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

        return fig

    def run_full_analysis(self):
        """Run complete Monte Carlo analysis for all time periods"""
        print("="*70)
        print("MONTE CARLO STRATEGY PERFORMANCE SIMULATION")
        print("="*70)
        print(f"\nSimulation Parameters:")
        print(f"  Market Annual Return:          {self.params.market_annual_return*100:.1f}%")
        print(f"  Market Annual Volatility:      {self.params.market_annual_vol*100:.1f}%")
        print(f"  Strategy Information Ratio:    {self.params.strategy_info_ratio:.2f}")
        print(f"  Strategy Tracking Error:       {self.params.strategy_tracking_error*100:.1f}%")
        print(f"  Number of Simulations:         {self.params.n_simulations:,}")
        print(f"  Time Periods:                  {self.params.periods} years")

        all_results = {}
        all_summaries = {}

        for period in self.params.periods:
            # Run simulations
            results = self.run_simulations(period)
            all_results[period] = results

            # Analyze and print results
            summary = self.analyze_results(results, period)
            all_summaries[period] = summary
            self.print_summary(summary)

        # Create visualizations
        self.create_visualizations(all_results)

        # Create comparison table
        self.create_comparison_table(all_summaries)

        return all_results, all_summaries

    def create_comparison_table(self, summaries: Dict[int, Dict]):
        """Create a comparison table across all time periods"""
        print(f"\n{'='*70}")
        print("SUMMARY COMPARISON ACROSS TIME PERIODS")
        print(f"{'='*70}\n")

        # Create DataFrame for easier formatting
        comparison_data = {
            'Period': [],
            'Mean IR': [],
            'Mean Rel Perf (%)': [],
            'Mean Beta-Adj (%)': [],
            'Outperform Rate (%)': []
        }

        for period in sorted(summaries.keys()):
            s = summaries[period]
            comparison_data['Period'].append(f'{period}Y')
            comparison_data['Mean IR'].append(f"{s['ir_mean']:.3f}")
            comparison_data['Mean Rel Perf (%)'].append(f"{s['rel_perf_mean']:.2f}")
            comparison_data['Mean Beta-Adj (%)'].append(f"{s['beta_adj_mean']:.2f}")
            comparison_data['Outperform Rate (%)'].append(f"{s['outperformance_rate']:.1f}")

        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        print()


def main():
    """Main execution function"""
    # Create simulation parameters
    params = SimulationParams(
        market_annual_return=0.08,      # 8% annual return
        market_annual_vol=0.16,         # 16% annual volatility
        strategy_info_ratio=0.5,        # IR of 0.5 (moderate skill)
        strategy_tracking_error=0.05,   # 5% tracking error
        n_simulations=10000,            # 10,000 simulations
        periods=[1, 3, 5, 10]           # Analyze 1, 3, 5, and 10 year periods
    )

    # Create simulator
    simulator = MonteCarloSimulator(params)

    # Run full analysis
    results, summaries = simulator.run_full_analysis()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
