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
from mpl_toolkits.mplot3d import Axes3D
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

    # Information ratios to test (for multi-IR analysis)
    info_ratios: List[float] = None

    # Tracking errors to test (for multi-TE analysis)
    tracking_errors: List[float] = None

    def __post_init__(self):
        if self.periods is None:
            self.periods = [1, 3, 5, 10]
        if self.info_ratios is None:
            self.info_ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
        if self.tracking_errors is None:
            self.tracking_errors = [0.01, 0.02, 0.03, 0.04, 0.05]


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

    def run_simulations(self, period_years: int, verbose: bool = True) -> List[PerformanceMetrics]:
        """
        Run multiple Monte Carlo simulations for a given time period

        Args:
            period_years: Time period in years
            verbose: If True, print progress messages

        Returns:
            List of PerformanceMetrics for each simulation
        """
        n_days = period_years * self.params.trading_days_per_year
        results = []

        if verbose:
            print(f"\nRunning {self.params.n_simulations} simulations for {period_years} year(s)...")

        for i in range(self.params.n_simulations):
            if verbose and (i + 1) % 1000 == 0:
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

    def run_multi_ir_analysis(self):
        """
        Run Monte Carlo analysis across multiple Information Ratios and time periods

        Returns:
            Dictionary with structure: {ir_value: {period: summary}}
        """
        print("="*80)
        print("MULTI-DIMENSIONAL MONTE CARLO ANALYSIS")
        print("Varying Information Ratio and Time Period")
        print("="*80)
        print(f"\nSimulation Parameters:")
        print(f"  Market Annual Return:          {self.params.market_annual_return*100:.1f}%")
        print(f"  Market Annual Volatility:      {self.params.market_annual_vol*100:.1f}%")
        print(f"  Strategy Tracking Error:       {self.params.strategy_tracking_error*100:.1f}%")
        print(f"  Number of Simulations:         {self.params.n_simulations:,}")
        print(f"  Information Ratios:            {self.params.info_ratios}")
        print(f"  Time Periods:                  {self.params.periods} years")

        all_results = {}  # {ir: {period: [PerformanceMetrics]}}

        total_runs = len(self.params.info_ratios) * len(self.params.periods)
        current_run = 0

        for ir in self.params.info_ratios:
            print(f"\n{'='*80}")
            print(f"TESTING INFORMATION RATIO: {ir:.2f}")
            print(f"{'='*80}")

            # Update the IR parameter
            self.params.strategy_info_ratio = ir
            all_results[ir] = {}

            for period in self.params.periods:
                current_run += 1
                print(f"[Progress: {current_run}/{total_runs}] Running simulations for IR={ir:.2f}, Period={period}Y... ", end='', flush=True)

                # Run simulations (suppress detailed progress)
                results = self.run_simulations(period, verbose=False)
                all_results[ir][period] = results

                # Analyze results
                summary = self.analyze_results(results, period)
                print(f"→ Outperformance Rate: {summary['outperformance_rate']:.1f}%")

        # Create matrix and visualizations
        self.create_outperformance_matrix(all_results)
        self.create_multi_ir_visualizations(all_results)

        return all_results

    def create_outperformance_matrix(self, all_results: Dict[float, Dict[int, List[PerformanceMetrics]]]):
        """
        Create and display a matrix of outperformance rates across IR and time periods

        Args:
            all_results: Dictionary {ir_value: {period: [PerformanceMetrics]}}
        """
        print(f"\n{'='*80}")
        print("OUTPERFORMANCE RATE MATRIX")
        print("(Percentage of simulations where strategy beats market)")
        print(f"{'='*80}\n")

        # Prepare data for matrix
        ir_values = sorted(all_results.keys())
        periods = sorted(self.params.periods)

        # Build matrix data
        matrix_data = []
        for ir in ir_values:
            row_data = {'IR': f'{ir:.2f}'}
            for period in periods:
                results = all_results[ir][period]
                strategy_rets = np.array([r.strategy_return for r in results])
                market_rets = np.array([r.market_return for r in results])
                outperf_rate = np.sum(strategy_rets > market_rets) / len(results) * 100
                row_data[f'{period}Y'] = f'{outperf_rate:.1f}%'
            matrix_data.append(row_data)

        # Create DataFrame
        df = pd.DataFrame(matrix_data)

        # Display the matrix
        print(df.to_string(index=False))
        print()

        # Also create numeric version for heatmap
        numeric_matrix = np.zeros((len(ir_values), len(periods)))
        for i, ir in enumerate(ir_values):
            for j, period in enumerate(periods):
                results = all_results[ir][period]
                strategy_rets = np.array([r.strategy_return for r in results])
                market_rets = np.array([r.market_return for r in results])
                numeric_matrix[i, j] = np.sum(strategy_rets > market_rets) / len(results) * 100

        return df, numeric_matrix, ir_values, periods

    def create_multi_ir_visualizations(self, all_results: Dict[float, Dict[int, List[PerformanceMetrics]]],
                                      save_path: str = 'monte_carlo_multi_ir_results.png'):
        """
        Create visualizations for multi-IR analysis including heatmap

        Args:
            all_results: Dictionary {ir_value: {period: [PerformanceMetrics]}}
            save_path: Path to save the figure
        """
        # Get matrix data
        df, matrix, ir_values, periods = self.create_outperformance_matrix(all_results)

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Plot 1: Heatmap of Outperformance Rates (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :])
        im = ax1.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=40, vmax=100)

        # Set ticks
        ax1.set_xticks(np.arange(len(periods)))
        ax1.set_yticks(np.arange(len(ir_values)))
        ax1.set_xticklabels([f'{p}Y' for p in periods])
        ax1.set_yticklabels([f'{ir:.2f}' for ir in ir_values])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Outperformance Rate (%)', rotation=270, labelpad=20, fontsize=12)

        # Add text annotations
        for i in range(len(ir_values)):
            for j in range(len(periods)):
                text = ax1.text(j, i, f'{matrix[i, j]:.1f}%',
                               ha="center", va="center", color="black", fontsize=10, fontweight='bold')

        ax1.set_xlabel('Time Period', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Information Ratio', fontsize=14, fontweight='bold')
        ax1.set_title('Strategy Outperformance Rate Heatmap\n(Information Ratio × Time Period)',
                     fontsize=16, fontweight='bold', pad=20)

        # Plot 2: Outperformance rate vs IR for different periods
        ax2 = fig.add_subplot(gs[1, 0])
        colors = plt.cm.viridis(np.linspace(0, 1, len(periods)))
        for j, (period, color) in enumerate(zip(periods, colors)):
            rates = [matrix[i, j] for i in range(len(ir_values))]
            ax2.plot(ir_values, rates, marker='o', linewidth=2, label=f'{period}Y', color=color)

        ax2.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% (Random)')
        ax2.set_xlabel('Information Ratio', fontsize=12)
        ax2.set_ylabel('Outperformance Rate (%)', fontsize=12)
        ax2.set_title('Outperformance Rate vs Information Ratio', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(40, 105)

        # Plot 3: Outperformance rate vs Period for different IRs
        ax3 = fig.add_subplot(gs[1, 1])

        # Select a subset of IRs to plot for clarity (adaptively based on available IRs)
        n_irs = len(ir_values)
        if n_irs >= 8:
            selected_ir_indices = [0, 2, 4, 6, 7]  # For full 8 IRs
        elif n_irs >= 5:
            selected_ir_indices = [0, n_irs//2, n_irs-1]
        else:
            selected_ir_indices = list(range(n_irs))  # Use all if few IRs

        colors_ir = plt.cm.plasma(np.linspace(0, 1, len(selected_ir_indices)))

        for idx, (ir_idx, color) in enumerate(zip(selected_ir_indices, colors_ir)):
            ir = ir_values[ir_idx]
            rates = matrix[ir_idx, :]
            ax3.plot(periods, rates, marker='o', linewidth=2, label=f'IR={ir:.2f}', color=color)

        ax3.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% (Random)')
        ax3.set_xlabel('Time Period (Years)', fontsize=12)
        ax3.set_ylabel('Outperformance Rate (%)', fontsize=12)
        ax3.set_title('Outperformance Rate vs Time Period', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(40, 105)

        # Plot 4: Distribution of IRs for different target IRs (1 year)
        ax4 = fig.add_subplot(gs[2, 0])

        # Select IRs that exist in the results
        candidate_irs = [0.1, 0.3, 0.5, 1.0]
        selected_irs = [ir for ir in candidate_irs if ir in all_results]

        # If none of the candidates exist, use some that do
        if not selected_irs:
            selected_irs = sorted(list(all_results.keys()))[:4]

        colors_dist = plt.cm.coolwarm(np.linspace(0, 1, len(selected_irs)))

        for target_ir, color in zip(selected_irs, colors_dist):
            if target_ir in all_results and 1 in all_results[target_ir]:
                results = all_results[target_ir][1]  # 1 year
                realized_irs = [r.information_ratio for r in results]
                ax4.hist(realized_irs, bins=30, alpha=0.5, label=f'Target IR={target_ir:.2f}',
                        color=color, density=True)

        ax4.set_xlabel('Realized Information Ratio', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.set_title('Realized IR Distribution (1 Year Period)', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Plot 5: Mean relative performance by IR and period
        ax5 = fig.add_subplot(gs[2, 1])

        # Calculate mean relative performance
        mean_rel_perf = np.zeros((len(ir_values), len(periods)))
        for i, ir in enumerate(ir_values):
            for j, period in enumerate(periods):
                results = all_results[ir][period]
                rel_perfs = [r.relative_performance * 100 for r in results]
                mean_rel_perf[i, j] = np.mean(rel_perfs)

        # Plot lines for each IR
        for i, (ir, color) in enumerate(zip(ir_values, plt.cm.viridis(np.linspace(0, 1, len(ir_values))))):
            ax5.plot(periods, mean_rel_perf[i, :], marker='o', linewidth=2,
                    label=f'IR={ir:.2f}', color=color, alpha=0.7)

        ax5.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax5.set_xlabel('Time Period (Years)', fontsize=12)
        ax5.set_ylabel('Mean Relative Performance (%)', fontsize=12)
        ax5.set_title('Mean Excess Return vs Period', fontsize=14, fontweight='bold')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nMulti-IR visualization saved to: {save_path}")

        return fig

    def run_full_3d_analysis(self):
        """
        Run Monte Carlo analysis across Information Ratios, Time Periods, AND Tracking Errors

        Returns:
            Dictionary with structure: {te: {ir: {period: [PerformanceMetrics]}}}
        """
        print("="*80)
        print("3D MONTE CARLO ANALYSIS")
        print("Varying Information Ratio × Time Period × Tracking Error")
        print("="*80)
        print(f"\nSimulation Parameters:")
        print(f"  Market Annual Return:          {self.params.market_annual_return*100:.1f}%")
        print(f"  Market Annual Volatility:      {self.params.market_annual_vol*100:.1f}%")
        print(f"  Number of Simulations:         {self.params.n_simulations:,}")
        print(f"  Information Ratios:            {self.params.info_ratios}")
        print(f"  Time Periods:                  {self.params.periods} years")
        print(f"  Tracking Errors:               {[f'{te*100:.0f}%' for te in self.params.tracking_errors]}")

        total_runs = len(self.params.tracking_errors) * len(self.params.info_ratios) * len(self.params.periods)
        print(f"\nTotal combinations: {total_runs:,} ({len(self.params.tracking_errors)} TEs × {len(self.params.info_ratios)} IRs × {len(self.params.periods)} periods)")
        print(f"Total simulations: {total_runs * self.params.n_simulations:,}\n")

        all_results = {}  # {te: {ir: {period: [PerformanceMetrics]}}}
        current_run = 0

        for te in self.params.tracking_errors:
            print(f"\n{'='*80}")
            print(f"TESTING TRACKING ERROR: {te*100:.0f}%")
            print(f"{'='*80}")

            # Update the TE parameter
            self.params.strategy_tracking_error = te
            all_results[te] = {}

            for ir in self.params.info_ratios:
                # Update the IR parameter
                self.params.strategy_info_ratio = ir
                all_results[te][ir] = {}

                for period in self.params.periods:
                    current_run += 1
                    print(f"[{current_run}/{total_runs}] TE={te*100:.0f}%, IR={ir:.2f}, Period={period}Y... ",
                          end='', flush=True)

                    # Run simulations (suppress detailed progress)
                    results = self.run_simulations(period, verbose=False)
                    all_results[te][ir][period] = results

                    # Calculate outperformance rate
                    strategy_rets = np.array([r.strategy_return for r in results])
                    market_rets = np.array([r.market_return for r in results])
                    outperf_rate = np.sum(strategy_rets > market_rets) / len(results) * 100
                    print(f"→ {outperf_rate:.1f}%")

        # Create visualizations
        self.create_3d_visualizations(all_results)
        self.print_3d_summary(all_results)

        return all_results

    def print_3d_summary(self, all_results):
        """Print summary tables for each tracking error level"""
        print(f"\n{'='*80}")
        print("OUTPERFORMANCE RATE MATRICES BY TRACKING ERROR")
        print(f"{'='*80}\n")

        for te in sorted(all_results.keys()):
            print(f"\n{'─'*80}")
            print(f"Tracking Error: {te*100:.0f}%")
            print(f"{'─'*80}")

            # Build matrix for this TE
            ir_values = sorted(all_results[te].keys())
            periods = sorted(self.params.periods)

            matrix_data = []
            for ir in ir_values:
                row_data = {'IR': f'{ir:.2f}'}
                for period in periods:
                    results = all_results[te][ir][period]
                    strategy_rets = np.array([r.strategy_return for r in results])
                    market_rets = np.array([r.market_return for r in results])
                    outperf_rate = np.sum(strategy_rets > market_rets) / len(results) * 100
                    row_data[f'{period}Y'] = f'{outperf_rate:.1f}%'
                matrix_data.append(row_data)

            df = pd.DataFrame(matrix_data)
            print(df.to_string(index=False))
            print()

    def create_3d_visualizations(self, all_results,
                                 save_path: str = 'monte_carlo_3d_results.png'):
        """
        Create comprehensive visualizations for 3D analysis (TE × IR × Period)

        Args:
            all_results: Dictionary {te: {ir: {period: [PerformanceMetrics]}}}
            save_path: Path to save the figure
        """
        te_values = sorted(all_results.keys())
        ir_values = sorted(all_results[te_values[0]].keys())
        periods = sorted(self.params.periods)

        # Create large figure with multiple subplots
        n_te = len(te_values)
        fig = plt.figure(figsize=(20, 4 * n_te + 6))

        # Use gridspec for flexible layout
        gs = fig.add_gridspec(n_te + 2, 4, hspace=0.4, wspace=0.3,
                             height_ratios=[1] * n_te + [1.2, 1.2])

        # Top section: Heatmaps for each tracking error
        for te_idx, te in enumerate(te_values):
            ax = fig.add_subplot(gs[te_idx, :])

            # Build matrix
            matrix = np.zeros((len(ir_values), len(periods)))
            for i, ir in enumerate(ir_values):
                for j, period in enumerate(periods):
                    results = all_results[te][ir][period]
                    strategy_rets = np.array([r.strategy_return for r in results])
                    market_rets = np.array([r.market_return for r in results])
                    matrix[i, j] = np.sum(strategy_rets > market_rets) / len(results) * 100

            # Create heatmap
            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=45, vmax=100)

            # Set ticks
            ax.set_xticks(np.arange(len(periods)))
            ax.set_yticks(np.arange(len(ir_values)))
            ax.set_xticklabels([f'{p}Y' for p in periods], fontsize=10)
            ax.set_yticklabels([f'{ir:.2f}' for ir in ir_values], fontsize=9)

            # Add text annotations
            for i in range(len(ir_values)):
                for j in range(len(periods)):
                    text = ax.text(j, i, f'{matrix[i, j]:.0f}%',
                                 ha="center", va="center", color="black",
                                 fontsize=9, fontweight='bold')

            # Add colorbar on the right
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Win Rate (%)', rotation=270, labelpad=15, fontsize=10)

            ax.set_xlabel('Time Period', fontsize=11, fontweight='bold')
            ax.set_ylabel('Information Ratio', fontsize=11, fontweight='bold')
            ax.set_title(f'Tracking Error = {te*100:.0f}% | Outperformance Rate Matrix',
                        fontsize=12, fontweight='bold', pad=10)

        # Bottom section: Comparative analysis plots

        # Plot 1: Effect of Tracking Error on Outperformance (for selected IR/Period combos)
        ax_bottom1 = fig.add_subplot(gs[n_te, :2])

        # Select representative IR/Period combinations
        test_configs = [
            (0.3, 1, 'IR=0.3, 1Y'),
            (0.5, 5, 'IR=0.5, 5Y'),
            (0.75, 10, 'IR=0.75, 10Y'),
            (1.0, 10, 'IR=1.0, 10Y')
        ]

        colors_line = plt.cm.tab10(np.arange(len(test_configs)))

        for (ir, period, label), color in zip(test_configs, colors_line):
            if ir in ir_values and period in periods:
                rates = []
                for te in te_values:
                    results = all_results[te][ir][period]
                    strategy_rets = np.array([r.strategy_return for r in results])
                    market_rets = np.array([r.market_return for r in results])
                    rate = np.sum(strategy_rets > market_rets) / len(results) * 100
                    rates.append(rate)

                ax_bottom1.plot([te*100 for te in te_values], rates, marker='o',
                              linewidth=2.5, label=label, color=color, markersize=8)

        ax_bottom1.axhline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_bottom1.set_xlabel('Tracking Error (%)', fontsize=12, fontweight='bold')
        ax_bottom1.set_ylabel('Outperformance Rate (%)', fontsize=12, fontweight='bold')
        ax_bottom1.set_title('Impact of Tracking Error on Win Rate\n(Selected Configurations)',
                            fontsize=13, fontweight='bold')
        ax_bottom1.legend(loc='best', fontsize=10)
        ax_bottom1.grid(True, alpha=0.3)
        ax_bottom1.set_ylim(45, 105)

        # Plot 2: 3D scatter showing relationship between TE, IR, and Win Rate (10Y period)
        ax_bottom2 = fig.add_subplot(gs[n_te, 2:], projection='3d')

        # Collect data for 10Y period
        if 10 in periods:
            te_scatter = []
            ir_scatter = []
            rate_scatter = []

            for te in te_values:
                for ir in ir_values:
                    results = all_results[te][ir][10]
                    strategy_rets = np.array([r.strategy_return for r in results])
                    market_rets = np.array([r.market_return for r in results])
                    rate = np.sum(strategy_rets > market_rets) / len(results) * 100

                    te_scatter.append(te * 100)
                    ir_scatter.append(ir)
                    rate_scatter.append(rate)

            # Create scatter plot with color mapping
            scatter = ax_bottom2.scatter(te_scatter, ir_scatter, rate_scatter,
                                        c=rate_scatter, cmap='RdYlGn', s=100,
                                        vmin=45, vmax=100, alpha=0.8, edgecolors='black')

            ax_bottom2.set_xlabel('Tracking Error (%)', fontsize=10, fontweight='bold', labelpad=8)
            ax_bottom2.set_ylabel('Information Ratio', fontsize=10, fontweight='bold', labelpad=8)
            ax_bottom2.set_zlabel('Win Rate (%)', fontsize=10, fontweight='bold', labelpad=8)
            ax_bottom2.set_title('3D View: TE × IR → Win Rate\n(10 Year Period)',
                                fontsize=12, fontweight='bold', pad=15)

            cbar = plt.colorbar(scatter, ax=ax_bottom2, shrink=0.6, pad=0.1)
            cbar.set_label('Win Rate (%)', rotation=270, labelpad=15, fontsize=9)

        # Plot 3: Tracking Error impact across all IRs (averaged across periods)
        ax_bottom3 = fig.add_subplot(gs[n_te + 1, :2])

        colors_ir = plt.cm.viridis(np.linspace(0, 1, len(ir_values)))

        for ir, color in zip(ir_values, colors_ir):
            avg_rates = []
            for te in te_values:
                # Average across all periods for this TE/IR combo
                rates_for_te = []
                for period in periods:
                    results = all_results[te][ir][period]
                    strategy_rets = np.array([r.strategy_return for r in results])
                    market_rets = np.array([r.market_return for r in results])
                    rate = np.sum(strategy_rets > market_rets) / len(results) * 100
                    rates_for_te.append(rate)
                avg_rates.append(np.mean(rates_for_te))

            ax_bottom3.plot([te*100 for te in te_values], avg_rates, marker='o',
                          linewidth=2, label=f'IR={ir:.2f}', color=color, alpha=0.8)

        ax_bottom3.axhline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_bottom3.set_xlabel('Tracking Error (%)', fontsize=12, fontweight='bold')
        ax_bottom3.set_ylabel('Avg Win Rate (%) Across All Periods', fontsize=12, fontweight='bold')
        ax_bottom3.set_title('Average Outperformance Rate vs Tracking Error\n(All Time Periods)',
                            fontsize=13, fontweight='bold')
        ax_bottom3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        ax_bottom3.grid(True, alpha=0.3)

        # Plot 4: Heatmap showing optimal TE for each IR/Period combo
        ax_bottom4 = fig.add_subplot(gs[n_te + 1, 2:])

        # Find best TE for each IR/Period combination
        best_te_matrix = np.zeros((len(ir_values), len(periods)))
        best_rate_matrix = np.zeros((len(ir_values), len(periods)))

        for i, ir in enumerate(ir_values):
            for j, period in enumerate(periods):
                best_rate = 0
                best_te = te_values[0]

                for te in te_values:
                    results = all_results[te][ir][period]
                    strategy_rets = np.array([r.strategy_return for r in results])
                    market_rets = np.array([r.market_return for r in results])
                    rate = np.sum(strategy_rets > market_rets) / len(results) * 100

                    if rate > best_rate:
                        best_rate = rate
                        best_te = te

                best_te_matrix[i, j] = best_te * 100
                best_rate_matrix[i, j] = best_rate

        im = ax_bottom4.imshow(best_te_matrix, cmap='viridis', aspect='auto')

        ax_bottom4.set_xticks(np.arange(len(periods)))
        ax_bottom4.set_yticks(np.arange(len(ir_values)))
        ax_bottom4.set_xticklabels([f'{p}Y' for p in periods], fontsize=10)
        ax_bottom4.set_yticklabels([f'{ir:.2f}' for ir in ir_values], fontsize=9)

        # Annotate with optimal TE and win rate
        for i in range(len(ir_values)):
            for j in range(len(periods)):
                text = ax_bottom4.text(j, i,
                                      f'{best_te_matrix[i, j]:.0f}%\n({best_rate_matrix[i, j]:.0f}%)',
                                      ha="center", va="center", color="white",
                                      fontsize=8, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax_bottom4)
        cbar.set_label('Optimal TE (%)', rotation=270, labelpad=15, fontsize=10)

        ax_bottom4.set_xlabel('Time Period', fontsize=11, fontweight='bold')
        ax_bottom4.set_ylabel('Information Ratio', fontsize=11, fontweight='bold')
        ax_bottom4.set_title('Optimal Tracking Error for Each Configuration\n(Value shows TE% and Win Rate%)',
                            fontsize=12, fontweight='bold', pad=10)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n3D visualization saved to: {save_path}")

        return fig


def main():
    """Main execution function"""
    # Create simulation parameters
    params = SimulationParams(
        market_annual_return=0.08,      # 8% annual return
        market_annual_vol=0.16,         # 16% annual volatility
        strategy_info_ratio=0.5,        # Base IR (will be varied)
        strategy_tracking_error=0.05,   # Base TE (will be varied)
        n_simulations=10000,            # 10,000 simulations per combination
        periods=[1, 3, 5, 10],          # Analyze 1, 3, 5, and 10 year periods
        info_ratios=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],  # IR values to test
        tracking_errors=[0.01, 0.02, 0.03, 0.04, 0.05]  # TE values to test
    )

    # Create simulator
    simulator = MonteCarloSimulator(params)

    # Run full 3D analysis (TE × IR × Period)
    results = simulator.run_full_3d_analysis()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved:")
    print("  - monte_carlo_3d_results.png (comprehensive 3D visualizations)")


if __name__ == "__main__":
    main()
