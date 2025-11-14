"""Time series LLM model for financial forecasting"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesLLM(nn.Module):
    """
    Time Series LLM that adapts language models for sequential prediction.

    Uses a pretrained LLM backbone with custom input/output layers for
    numerical time series data.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        d_input: int = 1,
        d_output: int = 1,
        hidden_dim: int = 768,
        use_pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Initialize Time Series LLM.

        Args:
            model_name: Base LLM model name
            d_input: Input dimension (number of features)
            d_output: Output dimension (prediction targets)
            hidden_dim: Hidden dimension (must match LLM hidden size)
            use_pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze LLM backbone
        """
        super().__init__()

        self.model_name = model_name
        self.d_input = d_input
        self.d_output = d_output
        self.hidden_dim = hidden_dim

        logger.info(f"Initializing TimeSeriesLLM with {model_name}")

        # Load base LLM
        if use_pretrained:
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModelForCausalLM.from_config(config)

        # Get actual hidden size from model
        self.hidden_dim = self.backbone.config.hidden_size

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")

        # Input projection: map time series features to LLM hidden dim
        self.input_projection = nn.Linear(d_input, self.hidden_dim)

        # Output projection: map LLM hidden dim to predictions
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, d_output),
        )

        logger.info(f"Model initialized with hidden_dim={self.hidden_dim}")

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_input)
            attention_mask: Attention mask
            output_hidden_states: Whether to output hidden states

        Returns:
            Dictionary with predictions and optional hidden states
        """
        batch_size, seq_len, _ = x.shape

        # Project input to hidden dimension
        x = self.input_projection(x)  # (B, S, H)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=x.device)

        # Pass through backbone
        # We need to use the model's embedding dimension
        outputs = self.backbone.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        ) if hasattr(self.backbone, 'transformer') else self.backbone.model(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state  # (B, S, H)

        # Project to output dimension
        predictions = self.output_projection(hidden_states)  # (B, S, d_output)

        result = {
            "predictions": predictions,
            "last_hidden_state": hidden_states,
        }

        if output_hidden_states:
            result["hidden_states"] = outputs.hidden_states

        return result

    def predict_future(
        self,
        x: torch.Tensor,
        steps: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict future values autoregressively.

        Args:
            x: Input sequence (batch_size, sequence_length, d_input)
            steps: Number of steps to predict
            temperature: Sampling temperature (not used for deterministic prediction)

        Returns:
            Predictions (batch_size, steps, d_output)
        """
        self.eval()
        predictions = []

        with torch.no_grad():
            current_seq = x.clone()

            for _ in range(steps):
                # Predict next step
                output = self.forward(current_seq)
                next_pred = output["predictions"][:, -1:, :]  # (B, 1, d_output)

                predictions.append(next_pred)

                # Update sequence (use prediction as next input)
                # If d_output != d_input, we need to handle this
                if self.d_output == self.d_input:
                    current_seq = torch.cat([current_seq[:, 1:, :], next_pred], dim=1)
                else:
                    # Pad or project as needed
                    if self.d_output < self.d_input:
                        # Pad with zeros
                        padding = torch.zeros(
                            next_pred.shape[0], 1, self.d_input - self.d_output,
                            device=next_pred.device
                        )
                        next_input = torch.cat([next_pred, padding], dim=-1)
                    else:
                        # Take first d_input dimensions
                        next_input = next_pred[:, :, :self.d_input]

                    current_seq = torch.cat([current_seq[:, 1:, :], next_input], dim=1)

        predictions = torch.cat(predictions, dim=1)  # (B, steps, d_output)
        return predictions

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            predictions: Model predictions
            targets: Target values
            mask: Optional mask for valid positions

        Returns:
            Loss value
        """
        loss = nn.functional.mse_loss(predictions, targets, reduction='none')

        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss


class AdaptiveTimeSeriesLLM(TimeSeriesLLM):
    """
    Enhanced Time Series LLM with adaptive components.

    Includes attention mechanisms and better handling of temporal patterns.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        d_input: int = 1,
        d_output: int = 1,
        use_temporal_encoding: bool = True,
        **kwargs
    ):
        """
        Initialize Adaptive Time Series LLM.

        Args:
            model_name: Base LLM model name
            d_input: Input dimension
            d_output: Output dimension
            use_temporal_encoding: Whether to use learnable temporal encodings
            **kwargs: Additional arguments for parent class
        """
        super().__init__(model_name, d_input, d_output, **kwargs)

        self.use_temporal_encoding = use_temporal_encoding

        if use_temporal_encoding:
            # Learnable temporal position encodings
            self.temporal_encoding = nn.Parameter(
                torch.randn(1, 512, self.hidden_dim) * 0.02
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with temporal encoding."""
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x)

        # Add temporal encoding
        if self.use_temporal_encoding:
            x = x + self.temporal_encoding[:, :seq_len, :]

        # Create attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=x.device)

        # Pass through backbone
        outputs = self.backbone.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        ) if hasattr(self.backbone, 'transformer') else self.backbone.model(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state

        # Project to output
        predictions = self.output_projection(hidden_states)

        result = {
            "predictions": predictions,
            "last_hidden_state": hidden_states,
        }

        if output_hidden_states:
            result["hidden_states"] = outputs.hidden_states

        return result
