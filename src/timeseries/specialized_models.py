"""
Specialized time series foundation models.

This module provides wrappers for models specifically designed for time series forecasting:
- Chronos (Amazon): T5-based time series foundation model
- Lag-Llama: Llama-based time series model
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChronosModel(nn.Module):
    """
    Wrapper for Amazon's Chronos time series foundation model.

    Chronos is a family of pretrained time series forecasting models based on T5.
    It tokenizes time series values and treats forecasting as a language modeling task.

    Available models:
    - chronos-t5-tiny: ~8M parameters
    - chronos-t5-mini: ~20M parameters
    - chronos-t5-small: ~46M parameters
    - chronos-t5-base: ~200M parameters
    - chronos-t5-large: ~710M parameters
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Chronos model.

        Args:
            model_size: Size of model ('tiny', 'mini', 'small', 'base', 'large')
            device: Device to load model on
            torch_dtype: Torch dtype for model
        """
        super().__init__()

        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

        # Model name mapping
        model_names = {
            "tiny": "amazon/chronos-t5-tiny",
            "mini": "amazon/chronos-t5-mini",
            "small": "amazon/chronos-t5-small",
            "base": "amazon/chronos-t5-base",
            "large": "amazon/chronos-t5-large",
        }

        if model_size not in model_names:
            raise ValueError(
                f"Invalid model_size: {model_size}. "
                f"Choose from: {list(model_names.keys())}"
            )

        self.model_name = model_names[model_size]
        logger.info(f"Loading Chronos model: {self.model_name}")

        # Load the model
        try:
            from chronos import ChronosPipeline

            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=self.torch_dtype,
            )

            logger.info(f"Chronos model loaded successfully on {self.device}")

        except ImportError:
            logger.error(
                "Chronos package not installed. Install with: "
                "pip install git+https://github.com/amazon-science/chronos-forecasting.git"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Chronos model: {e}")
            raise

    def predict(
        self,
        context: Union[torch.Tensor, np.ndarray, List],
        prediction_length: int,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate forecasts.

        Args:
            context: Historical time series data
            prediction_length: Number of steps to forecast
            num_samples: Number of sample trajectories
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling

        Returns:
            Forecasts of shape (num_samples, prediction_length)
        """
        # Convert to appropriate format
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context)
        elif isinstance(context, list):
            context = torch.tensor(context)

        # Ensure 1D
        if len(context.shape) > 1:
            context = context.squeeze()

        # Generate forecast
        forecast = self.pipeline.predict(
            context=context,
            prediction_length=prediction_length,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return forecast

    def forward(
        self,
        x: torch.Tensor,
        prediction_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass compatible with TimeSeriesTrainer.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            prediction_length: Number of steps to predict

        Returns:
            Dictionary with predictions
        """
        batch_size, seq_len, n_features = x.shape

        if prediction_length is None:
            prediction_length = 1

        # For multi-feature, we'll forecast each feature separately
        # and combine them
        all_predictions = []

        for feature_idx in range(n_features):
            feature_predictions = []

            for batch_idx in range(batch_size):
                # Extract context for this sample and feature
                context = x[batch_idx, :, feature_idx]

                # Generate forecast
                forecast = self.predict(
                    context=context,
                    prediction_length=prediction_length,
                    num_samples=1,
                )

                feature_predictions.append(forecast[0])  # Take first sample

            # Stack predictions for this feature
            feature_tensor = torch.stack(feature_predictions)
            all_predictions.append(feature_tensor)

        # Combine all features: (batch, pred_len, features)
        predictions = torch.stack(all_predictions, dim=-1)

        return {
            "predictions": predictions,
            "last_hidden_state": predictions,  # For compatibility
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss (for compatibility with trainer)."""
        loss = nn.functional.mse_loss(predictions, targets, reduction='none')

        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss


class LagLlamaModel(nn.Module):
    """
    Wrapper for Lag-Llama time series foundation model.

    Lag-Llama is a foundation model for time series forecasting based on
    the Llama architecture, trained on diverse time series data.
    """

    def __init__(
        self,
        model_path: str = "time-series-foundation-models/Lag-Llama",
        device: Optional[str] = None,
        context_length: int = 32,
        prediction_length: int = 1,
        use_rope_scaling: bool = True,
    ):
        """
        Initialize Lag-Llama model.

        Args:
            model_path: HuggingFace model path
            device: Device to load model on
            context_length: Length of context window
            prediction_length: Default prediction length
            use_rope_scaling: Use RoPE scaling for longer sequences
        """
        super().__init__()

        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.context_length = context_length
        self.prediction_length = prediction_length

        logger.info(f"Loading Lag-Llama model from: {model_path}")

        try:
            from gluonts.torch.model.lag_llama import LagLlamaLightningModule
            from gluonts.dataset.common import ListDataset

            # Load pretrained model
            self.model = LagLlamaLightningModule.load_from_checkpoint(
                model_path,
                map_location=self.device,
            )
            self.model.eval()

            logger.info(f"Lag-Llama model loaded successfully on {self.device}")

        except ImportError:
            logger.error(
                "GluonTS not installed. Install with: "
                "pip install gluonts[torch]"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Lag-Llama model: {e}")
            logger.info(
                "Note: Lag-Llama requires downloading the checkpoint. "
                "Visit: https://github.com/time-series-foundation-models/lag-llama"
            )
            raise

    def prepare_data(
        self,
        x: torch.Tensor,
    ) -> List[Dict]:
        """
        Prepare data in GluonTS format.

        Args:
            x: Input tensor (batch_size, sequence_length, features)

        Returns:
            List of data dictionaries
        """
        batch_size, seq_len, n_features = x.shape

        data_list = []
        for batch_idx in range(batch_size):
            for feature_idx in range(n_features):
                data_entry = {
                    "target": x[batch_idx, :, feature_idx].cpu().numpy(),
                    "start": 0,  # Placeholder
                }
                data_list.append(data_entry)

        return data_list

    def predict(
        self,
        context: Union[torch.Tensor, np.ndarray],
        prediction_length: Optional[int] = None,
        num_samples: int = 100,
    ) -> torch.Tensor:
        """
        Generate forecasts.

        Args:
            context: Historical context
            prediction_length: Steps to forecast
            num_samples: Number of samples

        Returns:
            Forecasts
        """
        if prediction_length is None:
            prediction_length = self.prediction_length

        # Convert to numpy if needed
        if isinstance(context, torch.Tensor):
            context = context.cpu().numpy()

        # Create dataset
        from gluonts.dataset.common import ListDataset

        dataset = ListDataset(
            [{"target": context, "start": 0}],
            freq="1H",  # Placeholder frequency
        )

        # Generate predictions
        forecasts = self.model.predict(
            dataset,
            num_samples=num_samples,
        )

        # Extract samples
        forecast_samples = next(iter(forecasts)).samples

        return torch.from_numpy(forecast_samples)

    def forward(
        self,
        x: torch.Tensor,
        prediction_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass compatible with TimeSeriesTrainer.

        Args:
            x: Input tensor (batch_size, sequence_length, features)
            prediction_length: Prediction horizon

        Returns:
            Dictionary with predictions
        """
        batch_size, seq_len, n_features = x.shape

        if prediction_length is None:
            prediction_length = self.prediction_length

        all_predictions = []

        for feature_idx in range(n_features):
            feature_predictions = []

            for batch_idx in range(batch_size):
                context = x[batch_idx, :, feature_idx]

                # Generate forecast
                forecast = self.predict(
                    context=context,
                    prediction_length=prediction_length,
                    num_samples=1,
                )

                # Take mean of samples
                feature_predictions.append(forecast.mean(0))

            feature_tensor = torch.stack(feature_predictions)
            all_predictions.append(feature_tensor)

        # Combine: (batch, pred_len, features)
        predictions = torch.stack(all_predictions, dim=-1)

        return {
            "predictions": predictions,
            "last_hidden_state": predictions,
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss."""
        loss = nn.functional.mse_loss(predictions, targets, reduction='none')

        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss


def load_specialized_model(
    model_name: str,
    model_size: Optional[str] = "small",
    **kwargs
) -> nn.Module:
    """
    Convenience function to load specialized time series models.

    Args:
        model_name: Model name ('chronos', 'lag-llama')
        model_size: Model size (for Chronos: 'tiny', 'small', 'base', 'large')
        **kwargs: Additional arguments for model

    Returns:
        Initialized model

    Examples:
        >>> # Load Chronos tiny model
        >>> model = load_specialized_model("chronos", model_size="tiny")

        >>> # Load Lag-Llama
        >>> model = load_specialized_model("lag-llama")
    """
    model_name = model_name.lower()

    if model_name == "chronos":
        return ChronosModel(model_size=model_size, **kwargs)
    elif model_name in ["lag-llama", "lagllama"]:
        return LagLlamaModel(**kwargs)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: 'chronos', 'lag-llama'"
        )


def get_model_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about available specialized time series models.

    Returns:
        Dictionary with model information
    """
    return {
        "chronos": {
            "name": "Chronos (Amazon)",
            "architecture": "T5-based",
            "sizes": "tiny (8M), mini (20M), small (46M), base (200M), large (710M)",
            "best_for": "General time series forecasting, zero-shot",
            "installation": "pip install git+https://github.com/amazon-science/chronos-forecasting.git",
            "paper": "https://arxiv.org/abs/2403.07815",
            "strengths": "Pre-trained on diverse data, handles multiple frequencies",
        },
        "lag-llama": {
            "name": "Lag-Llama",
            "architecture": "Llama-based",
            "sizes": "~1B parameters",
            "best_for": "Probabilistic forecasting, handling lags",
            "installation": "pip install gluonts[torch]",
            "paper": "https://arxiv.org/abs/2310.08278",
            "strengths": "Probabilistic forecasts, uncertainty quantification",
        },
    }


def print_model_comparison():
    """Print comparison of specialized time series models."""
    models = get_model_info()

    print("\n" + "=" * 80)
    print("SPECIALIZED TIME SERIES MODELS")
    print("=" * 80)

    for model_key, info in models.items():
        print(f"\n{info['name'].upper()} ({model_key})")
        print("-" * 80)
        print(f"  Architecture:  {info['architecture']}")
        print(f"  Sizes:         {info['sizes']}")
        print(f"  Best For:      {info['best_for']}")
        print(f"  Strengths:     {info['strengths']}")
        print(f"  Installation:  {info['installation']}")
        print(f"  Paper:         {info['paper']}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("  - Start with Chronos-tiny for quick experiments")
    print("  - Use Chronos-small/base for better accuracy")
    print("  - Use Lag-Llama for probabilistic forecasts with uncertainty")
    print("=" * 80 + "\n")
