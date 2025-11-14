"""LoRA (Low-Rank Adaptation) implementation and configuration"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA fine-tuning.

    Attributes:
        r: Rank of the low-rank matrices
        lora_alpha: Scaling factor for LoRA
        target_modules: List of module names to apply LoRA to
        lora_dropout: Dropout probability for LoRA layers
        bias: Bias training strategy ('none', 'all', 'lora_only')
        task_type: Type of task (default: CAUSAL_LM)
        inference_mode: Whether in inference mode
        modules_to_save: Additional modules to train fully
    """
    r: int = 8
    lora_alpha: int = 32
    target_modules: Optional[List[str]] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False
    modules_to_save: Optional[List[str]] = None

    def __post_init__(self):
        """Set default target modules if not specified."""
        if self.target_modules is None:
            # Common attention modules across different architectures
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj",      # MLP (LLaMA style)
            ]


class LoRAModuleSelector:
    """Helper class to select which modules to apply LoRA to."""

    # Predefined module groups for common architectures
    ATTENTION_MODULES = {
        "gpt2": ["c_attn", "c_proj"],
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "phi": ["q_proj", "k_proj", "v_proj", "dense"],
    }

    MLP_MODULES = {
        "gpt2": ["c_fc", "c_proj"],
        "llama": ["gate_proj", "up_proj", "down_proj"],
        "mistral": ["gate_proj", "up_proj", "down_proj"],
        "phi": ["fc1", "fc2"],
    }

    @classmethod
    def get_attention_modules(cls, model_type: str) -> List[str]:
        """Get attention module names for a model type."""
        return cls.ATTENTION_MODULES.get(model_type.lower(), cls.ATTENTION_MODULES["llama"])

    @classmethod
    def get_mlp_modules(cls, model_type: str) -> List[str]:
        """Get MLP module names for a model type."""
        return cls.MLP_MODULES.get(model_type.lower(), cls.MLP_MODULES["llama"])

    @classmethod
    def get_all_modules(cls, model_type: str) -> List[str]:
        """Get all applicable module names for a model type."""
        return cls.get_attention_modules(model_type) + cls.get_mlp_modules(model_type)

    @classmethod
    def get_custom_modules(
        cls,
        model_type: str,
        include_attention: bool = True,
        include_mlp: bool = False,
        custom_modules: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Get custom combination of modules.

        Args:
            model_type: Type of model architecture
            include_attention: Include attention modules
            include_mlp: Include MLP modules
            custom_modules: Additional custom module names

        Returns:
            List of module names to apply LoRA to
        """
        modules = []

        if include_attention:
            modules.extend(cls.get_attention_modules(model_type))

        if include_mlp:
            modules.extend(cls.get_mlp_modules(model_type))

        if custom_modules:
            modules.extend(custom_modules)

        return list(set(modules))  # Remove duplicates


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
    prepare_for_quantization: bool = True,
) -> nn.Module:
    """
    Apply LoRA to a model.

    Args:
        model: The model to apply LoRA to
        config: LoRA configuration
        prepare_for_quantization: Whether to prepare model for k-bit training

    Returns:
        PEFT model with LoRA applied
    """
    logger.info("Applying LoRA to model")
    logger.info(f"LoRA config: r={config.r}, alpha={config.lora_alpha}, "
                f"target_modules={config.target_modules}")

    # Prepare model for quantized training if needed
    if prepare_for_quantization:
        try:
            model = prepare_model_for_kbit_training(model)
            logger.info("Model prepared for k-bit training")
        except Exception as e:
            logger.warning(f"Could not prepare for k-bit training: {e}")

    # Convert to PEFT config
    peft_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=config.inference_mode,
        modules_to_save=config.modules_to_save,
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)

    logger.info("LoRA applied successfully")
    model.print_trainable_parameters()

    return model


def create_lora_config_for_model(
    model_type: str,
    r: int = 8,
    lora_alpha: int = 32,
    include_attention: bool = True,
    include_mlp: bool = False,
    lora_dropout: float = 0.1,
) -> LoRAConfig:
    """
    Create a LoRA config optimized for a specific model type.

    Args:
        model_type: Type of model (gpt2, llama, mistral, phi, etc.)
        r: LoRA rank
        lora_alpha: LoRA alpha
        include_attention: Apply LoRA to attention modules
        include_mlp: Apply LoRA to MLP modules
        lora_dropout: Dropout rate

    Returns:
        LoRAConfig instance
    """
    target_modules = LoRAModuleSelector.get_custom_modules(
        model_type=model_type,
        include_attention=include_attention,
        include_mlp=include_mlp,
    )

    return LoRAConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
    )
