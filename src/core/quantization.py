"""Quantization utilities and helpers"""

from typing import Optional, Dict, Any
import torch
from transformers import BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizationConfig:
    """Quantization configuration builder."""

    @staticmethod
    def get_4bit_config(
        compute_dtype: torch.dtype = torch.float16,
        use_double_quant: bool = True,
        quant_type: str = "nf4",
    ) -> BitsAndBytesConfig:
        """
        Get 4-bit quantization config.

        Args:
            compute_dtype: Computation dtype
            use_double_quant: Whether to use double quantization
            quant_type: Quantization type ('nf4' or 'fp4')

        Returns:
            BitsAndBytesConfig for 4-bit quantization
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_quant_type=quant_type,
        )

    @staticmethod
    def get_8bit_config() -> BitsAndBytesConfig:
        """
        Get 8-bit quantization config.

        Returns:
            BitsAndBytesConfig for 8-bit quantization
        """
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )

    @staticmethod
    def get_config(
        quantization_type: Optional[str],
        **kwargs
    ) -> Optional[BitsAndBytesConfig]:
        """
        Get quantization config based on type.

        Args:
            quantization_type: '4bit', '8bit', or None
            **kwargs: Additional configuration parameters

        Returns:
            BitsAndBytesConfig or None
        """
        if quantization_type == "4bit":
            return QuantizationConfig.get_4bit_config(**kwargs)
        elif quantization_type == "8bit":
            return QuantizationConfig.get_8bit_config()
        return None


class MemoryUtils:
    """Memory optimization utilities."""

    @staticmethod
    def print_memory_stats():
        """Print current GPU memory statistics."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(
                    f"GPU {i}: {allocated:.2f} GB allocated, "
                    f"{reserved:.2f} GB reserved"
                )
        else:
            logger.info("CUDA not available")

    @staticmethod
    def clear_cache():
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    @staticmethod
    def get_model_size(model) -> Dict[str, float]:
        """
        Calculate model size in memory.

        Args:
            model: PyTorch model

        Returns:
            Dictionary with size statistics in GB
        """
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = param_size + buffer_size

        return {
            "param_size_gb": param_size / 1024**3,
            "buffer_size_gb": buffer_size / 1024**3,
            "total_size_gb": total_size / 1024**3,
        }


def estimate_memory_requirements(
    num_parameters: int,
    bits: int = 16,
    batch_size: int = 1,
    sequence_length: int = 512,
) -> Dict[str, float]:
    """
    Estimate memory requirements for model training.

    Args:
        num_parameters: Number of model parameters
        bits: Quantization bits (4, 8, 16, or 32)
        batch_size: Training batch size
        sequence_length: Sequence length

    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_param = bits / 8

    # Model weights
    model_memory = num_parameters * bytes_per_param

    # Gradients (same size as model for full fine-tuning)
    gradient_memory = model_memory

    # Optimizer states (Adam: 2x model size for momentum and variance)
    optimizer_memory = model_memory * 2

    # Activations (rough estimate)
    activation_memory = (
        batch_size * sequence_length * num_parameters * 4 * bytes_per_param / 1000
    )

    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory

    return {
        "model_gb": model_memory / 1024**3,
        "gradients_gb": gradient_memory / 1024**3,
        "optimizer_gb": optimizer_memory / 1024**3,
        "activations_gb": activation_memory / 1024**3,
        "total_gb": total_memory / 1024**3,
    }


def print_quantization_info(model):
    """Print quantization information for a model."""
    logger.info("=" * 50)
    logger.info("Quantization Information")
    logger.info("=" * 50)

    quantized_layers = 0
    total_layers = 0

    for name, module in model.named_modules():
        total_layers += 1
        if hasattr(module, "weight") and module.weight is not None:
            dtype = module.weight.dtype
            if "int" in str(dtype).lower():
                quantized_layers += 1
                logger.info(f"Quantized layer: {name} ({dtype})")

    logger.info(f"\nTotal layers: {total_layers}")
    logger.info(f"Quantized layers: {quantized_layers}")
    logger.info("=" * 50)
