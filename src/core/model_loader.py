"""Model loading utilities with quantization support"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from typing import Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Unified model loader supporting various LLMs with optional quantization.

    Supports models like GPT-2, LLaMA, Mistral, Phi, etc.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir

    def load_model(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = False,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a pretrained language model with optional quantization.

        Args:
            model_name: HuggingFace model identifier or path
            quantization: Quantization type ('4bit', '8bit', or None)
            device_map: Device mapping strategy
            torch_dtype: Default dtype for model weights
            trust_remote_code: Whether to trust remote code

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {model_name}")

        # Configure quantization if requested
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization (NF4)")
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logger.info("Using 8-bit quantization")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=trust_remote_code,
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "cache_dir": self.cache_dir,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        logger.info(f"Model loaded successfully. Parameters: {self.count_parameters(model):,}")

        return model, tokenizer

    @staticmethod
    def count_parameters(model) -> int:
        """Count total parameters in model."""
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def count_trainable_parameters(model) -> Tuple[int, int]:
        """
        Count trainable parameters.

        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params

    def print_trainable_parameters(self, model):
        """Print trainable parameter statistics."""
        trainable, total = self.count_trainable_parameters(model)
        percentage = 100 * trainable / total
        logger.info(
            f"Trainable params: {trainable:,} || "
            f"Total params: {total:,} || "
            f"Trainable%: {percentage:.2f}%"
        )
