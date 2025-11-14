"""
Visualization utilities for time series forecasting.

Provides comprehensive plotting functions for comparing predictions
against realized values, analyzing errors, and visualizing model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


class ForecastVisualizer:
    """
    Comprehensive visualization for time series forecasts.

    Creates professional plots comparing predictions against realized values.
    """

    def __init__(self, output_dir: str = "./outputs/visualizations"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_predictions_vs_actual(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: Optional[np.ndarray] = None,
        title: str = "Predictions vs Actual Values",
        ylabel: str = "Value",
        save_name: Optional[str] = None,
        show_confidence: bool = False,
        confidence_lower: Optional[np.ndarray] = None,
        confidence_upper: Optional[np.ndarray] = None,
    ) -> str:
        """
        Plot predictions against actual values over time.

        Args:
            predictions: Predicted values
            actuals: Actual values
            dates: Date indices
            title: Plot title
            ylabel: Y-axis label
            save_name: Filename to save (without extension)
            show_confidence: Whether to show confidence intervals
            confidence_lower: Lower confidence bound
            confidence_upper: Upper confidence bound

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Handle multi-dimensional arrays
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        if len(actuals.shape) > 1:
            actuals = actuals.flatten()

        # Create x-axis
        if dates is not None:
            x = dates[-len(actuals):]
            if isinstance(x[0], (pd.Timestamp, np.datetime64)):
                x = pd.to_datetime(x)
        else:
            x = np.arange(len(actuals))

        # Plot main lines
        ax.plot(x, actuals, label='Actual', linewidth=2.5,
               color='#2E86AB', alpha=0.8, marker='o', markersize=3)
        ax.plot(x, predictions, label='Predicted', linewidth=2.5,
               color='#A23B72', alpha=0.8, marker='s', markersize=3)

        # Add confidence intervals if provided
        if show_confidence and confidence_lower is not None and confidence_upper is not None:
            if len(confidence_lower.shape) > 1:
                confidence_lower = confidence_lower.flatten()
            if len(confidence_upper.shape) > 1:
                confidence_upper = confidence_upper.flatten()

            ax.fill_between(x, confidence_lower, confidence_upper,
                          alpha=0.2, color='#A23B72', label='Confidence Interval')

        # Calculate and display metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        direction_acc = np.mean(np.sign(predictions) == np.sign(actuals))

        # Add metrics text box
        metrics_text = (
            f'MAE: {mae:.6f}\n'
            f'RMSE: {rmse:.6f}\n'
            f'Direction Acc: {direction_acc:.2%}'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', bbox=props)

        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Time' if dates is not None else 'Sample',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Rotate x-labels if dates
        if dates is not None and isinstance(x[0], (pd.Timestamp, np.datetime64)):
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Save
        if save_name is None:
            save_name = "predictions_vs_actual"
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
        plt.close()

        return str(save_path)

    def plot_forecast_horizon_analysis(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        horizon_names: Optional[List[str]] = None,
        title: str = "Forecast Accuracy by Horizon",
        save_name: Optional[str] = None,
    ) -> str:
        """
        Analyze forecast accuracy across different prediction horizons.

        Args:
            predictions: Predictions array (samples, horizons)
            actuals: Actuals array (samples, horizons)
            horizon_names: Names for each horizon
            title: Plot title
            save_name: Filename to save

        Returns:
            Path to saved plot
        """
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        if len(actuals.shape) == 1:
            actuals = actuals.reshape(-1, 1)

        n_horizons = predictions.shape[1]

        if horizon_names is None:
            horizon_names = [f"H+{i+1}" for i in range(n_horizons)]

        # Calculate metrics for each horizon
        metrics_data = {
            'Horizon': [],
            'MAE': [],
            'RMSE': [],
            'Direction Accuracy': [],
        }

        for h in range(n_horizons):
            pred_h = predictions[:, h]
            actual_h = actuals[:, h]

            mae = np.mean(np.abs(pred_h - actual_h))
            rmse = np.sqrt(np.mean((pred_h - actual_h) ** 2))
            dir_acc = np.mean(np.sign(pred_h) == np.sign(actual_h))

            metrics_data['Horizon'].append(horizon_names[h])
            metrics_data['MAE'].append(mae)
            metrics_data['RMSE'].append(rmse)
            metrics_data['Direction Accuracy'].append(dir_acc * 100)

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        # MAE
        axes[0].bar(metrics_data['Horizon'], metrics_data['MAE'],
                   color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[0].set_title('Mean Absolute Error', fontweight='bold')
        axes[0].set_ylabel('MAE', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # RMSE
        axes[1].bar(metrics_data['Horizon'], metrics_data['RMSE'],
                   color='#F18F01', alpha=0.7, edgecolor='black')
        axes[1].set_title('Root Mean Squared Error', fontweight='bold')
        axes[1].set_ylabel('RMSE', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        # Direction Accuracy
        axes[2].bar(metrics_data['Horizon'], metrics_data['Direction Accuracy'],
                   color='#06A77D', alpha=0.7, edgecolor='black')
        axes[2].axhline(50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
        axes[2].set_title('Direction Accuracy', fontweight='bold')
        axes[2].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save
        if save_name is None:
            save_name = "horizon_analysis"
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
        plt.close()

        return str(save_path)

    def plot_price_reconstruction(
        self,
        returns_predictions: np.ndarray,
        returns_actuals: np.ndarray,
        initial_price: float,
        dates: Optional[np.ndarray] = None,
        title: str = "Price Reconstruction from Returns",
        save_name: Optional[str] = None,
    ) -> str:
        """
        Reconstruct and plot prices from return predictions.

        Useful when predicting returns but want to see actual price movements.

        Args:
            returns_predictions: Predicted returns
            returns_actuals: Actual returns
            initial_price: Starting price for reconstruction
            dates: Date indices
            title: Plot title
            save_name: Filename to save

        Returns:
            Path to saved plot
        """
        # Flatten if needed
        if len(returns_predictions.shape) > 1:
            returns_predictions = returns_predictions.flatten()
        if len(returns_actuals.shape) > 1:
            returns_actuals = returns_actuals.flatten()

        # Reconstruct prices (assuming log returns)
        # Price_t = Price_0 * exp(sum(returns))
        cumulative_pred = np.cumsum(returns_predictions)
        cumulative_actual = np.cumsum(returns_actuals)

        price_pred = initial_price * np.exp(cumulative_pred)
        price_actual = initial_price * np.exp(cumulative_actual)

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Create x-axis
        if dates is not None:
            x = dates[-len(returns_actuals):]
            if isinstance(x[0], (pd.Timestamp, np.datetime64)):
                x = pd.to_datetime(x)
        else:
            x = np.arange(len(returns_actuals))

        # Plot 1: Reconstructed Prices
        axes[0].plot(x, price_actual, label='Actual Price',
                    linewidth=2.5, color='#2E86AB', alpha=0.8, marker='o', markersize=3)
        axes[0].plot(x, price_pred, label='Predicted Price',
                    linewidth=2.5, color='#A23B72', alpha=0.8, marker='s', markersize=3)
        axes[0].set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        axes[0].set_title('Reconstructed Price Paths', fontweight='bold')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Calculate price-based metrics
        price_mae = np.mean(np.abs(price_pred - price_actual))
        price_rmse = np.sqrt(np.mean((price_pred - price_actual) ** 2))
        price_mape = np.mean(np.abs((price_pred - price_actual) / price_actual)) * 100

        metrics_text = (
            f'Price MAE: ${price_mae:.2f}\n'
            f'Price RMSE: ${price_rmse:.2f}\n'
            f'Price MAPE: {price_mape:.2f}%'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[0].text(0.02, 0.98, metrics_text, transform=axes[0].transAxes,
                    fontsize=10, verticalalignment='top', bbox=props)

        # Plot 2: Cumulative Returns
        axes[1].plot(x, cumulative_actual * 100, label='Actual Cumulative Return',
                    linewidth=2.5, color='#2E86AB', alpha=0.8)
        axes[1].plot(x, cumulative_pred * 100, label='Predicted Cumulative Return',
                    linewidth=2.5, color='#A23B72', alpha=0.8)
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Time' if dates is not None else 'Sample',
                          fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Cumulative Returns', fontweight='bold')
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Rotate x-labels if dates
        if dates is not None and isinstance(x[0], (pd.Timestamp, np.datetime64)):
            for ax in axes:
                plt.sca(ax)
                plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Save
        if save_name is None:
            save_name = "price_reconstruction"
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
        plt.close()

        return str(save_path)

    def plot_comprehensive_comparison(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: Optional[np.ndarray] = None,
        title: str = "Comprehensive Forecast Analysis",
        save_name: Optional[str] = None,
    ) -> str:
        """
        Create a comprehensive 4-panel comparison plot.

        Args:
            predictions: Predicted values
            actuals: Actual values
            dates: Date indices
            title: Plot title
            save_name: Filename to save

        Returns:
            Path to saved plot
        """
        # Flatten if needed
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        if len(actuals.shape) > 1:
            actuals = actuals.flatten()

        errors = predictions - actuals

        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Create x-axis
        if dates is not None:
            x = dates[-len(actuals):]
            if isinstance(x[0], (pd.Timestamp, np.datetime64)):
                x = pd.to_datetime(x)
        else:
            x = np.arange(len(actuals))

        # Plot 1: Time series comparison
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(x, actuals, label='Actual', linewidth=2,
                color='#2E86AB', alpha=0.8, marker='o', markersize=2)
        ax1.plot(x, predictions, label='Predicted', linewidth=2,
                color='#A23B72', alpha=0.8, marker='s', markersize=2)
        ax1.fill_between(x, actuals, predictions, alpha=0.2, color='gray')
        ax1.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax1.set_title('Predictions vs Actual Over Time', fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

        if dates is not None and isinstance(x[0], (pd.Timestamp, np.datetime64)):
            plt.sca(ax1)
            plt.xticks(rotation=45, ha='right')

        # Plot 2: Scatter plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(actuals, predictions, alpha=0.5, s=30, color='#2E86AB')

        # Add perfect prediction line
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        ax2.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Perfect Prediction')

        # Calculate R²
        r2 = 1 - (np.sum((actuals - predictions) ** 2) /
                 np.sum((actuals - np.mean(actuals)) ** 2))

        ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax2.set_xlabel('Actual', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Predicted', fontsize=11, fontweight='bold')
        ax2.set_title('Scatter Plot', fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Error distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(errors, bins=40, edgecolor='black', alpha=0.7, color='#F18F01')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.axvline(np.mean(errors), color='green', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(errors):.6f}')
        ax3.set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('Error Distribution', fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Cumulative error
        ax4 = fig.add_subplot(gs[2, 0])
        cumulative_error = np.cumsum(np.abs(errors))
        ax4.plot(x, cumulative_error, linewidth=2, color='#06A77D')
        ax4.set_xlabel('Time' if dates is not None else 'Sample',
                      fontsize=11, fontweight='bold')
        ax4.set_ylabel('Cumulative |Error|', fontsize=11, fontweight='bold')
        ax4.set_title('Cumulative Absolute Error', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        if dates is not None and isinstance(x[0], (pd.Timestamp, np.datetime64)):
            plt.sca(ax4)
            plt.xticks(rotation=45, ha='right')

        # Plot 5: Metrics summary
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        # Calculate comprehensive metrics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        mape = np.mean(np.abs(errors / (np.abs(actuals) + 1e-8))) * 100
        direction_acc = np.mean(np.sign(predictions) == np.sign(actuals))
        median_ae = np.median(np.abs(errors))

        metrics_text = f"""
        PERFORMANCE METRICS
        {'='*40}

        Mean Absolute Error:        {mae:.6f}
        Root Mean Squared Error:    {rmse:.6f}
        Mean Abs. Percentage Error: {mape:.2f}%
        Median Absolute Error:      {median_ae:.6f}

        R² Score:                   {r2:.4f}
        Direction Accuracy:         {direction_acc:.2%}

        Error Statistics:
        - Mean Error:               {np.mean(errors):.6f}
        - Error Std Dev:            {np.std(errors):.6f}
        - Max Error:                {np.max(np.abs(errors)):.6f}
        """

        ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # Save
        if save_name is None:
            save_name = "comprehensive_comparison"
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved comprehensive plot to {save_path}")
        plt.close()

        return str(save_path)


def plot_multi_step_forecast(
    predictions: np.ndarray,
    actuals: np.ndarray,
    context: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
    output_dir: str = "./outputs/visualizations",
    title: str = "Multi-Step Forecast",
    save_name: str = "multi_step_forecast",
) -> str:
    """
    Visualize multi-step ahead forecasts with context.

    Args:
        predictions: Multi-step predictions
        actuals: Actual future values
        context: Historical context used for prediction
        dates: Date indices
        output_dir: Output directory
        title: Plot title
        save_name: Filename to save

    Returns:
        Path to saved plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Determine lengths
    n_context = len(context) if context is not None else 0
    n_forecast = len(predictions)

    # Create x-axis
    if dates is not None:
        x_context = dates[:n_context] if n_context > 0 else []
        x_forecast = dates[n_context:n_context + n_forecast]
    else:
        x_context = np.arange(n_context)
        x_forecast = np.arange(n_context, n_context + n_forecast)

    # Plot context if provided
    if context is not None and len(context) > 0:
        ax.plot(x_context, context, label='Historical Context',
               linewidth=2, color='gray', alpha=0.6, linestyle='-')

    # Plot forecast and actual
    ax.plot(x_forecast, actuals, label='Actual Future',
           linewidth=2.5, color='#2E86AB', alpha=0.8, marker='o', markersize=4)
    ax.plot(x_forecast, predictions, label='Predicted Future',
           linewidth=2.5, color='#A23B72', alpha=0.8, marker='s', markersize=4)

    # Add vertical line separating context from forecast
    if context is not None and len(context) > 0:
        ax.axvline(x_context[-1] if len(x_context) > 0 else 0,
                  color='red', linestyle='--', linewidth=2,
                  label='Forecast Start', alpha=0.7)

    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if dates is not None and len(dates) > 0:
        if isinstance(dates[0], (pd.Timestamp, np.datetime64)):
            plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    save_path = Path(output_dir) / f"{save_name}.png"
    plt.savefig(save_path, bbox_inches='tight')
    logger.info(f"Saved multi-step forecast plot to {save_path}")
    plt.close()

    return str(save_path)
