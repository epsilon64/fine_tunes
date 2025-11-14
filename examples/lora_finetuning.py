"""
LoRA fine-tuning example with quantization.

This example demonstrates efficient fine-tuning using LoRA and 4-bit quantization.
"""

import sys
sys.path.append("..")

from src.core.model_loader import ModelLoader
from src.core.lora import LoRAConfig, apply_lora, create_lora_config_for_model
from src.core.trainer import FineTuner
from src.datasets.base_loader import prepare_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run LoRA fine-tuning example."""

    logger.info("=" * 50)
    logger.info("LoRA Fine-Tuning Example")
    logger.info("=" * 50)

    # Configuration
    MODEL_NAME = "gpt2"
    DATASET_NAME = "wikitext"
    DATASET_SUBSET = "wikitext-2-raw-v1"
    OUTPUT_DIR = "./outputs/lora_finetuning"

    # LoRA settings
    LORA_R = 8
    LORA_ALPHA = 32
    USE_QUANTIZATION = False  # Set to True if you have CUDA available

    # 1. Load model with quantization
    logger.info("\n1. Loading model with quantization...")
    loader = ModelLoader()
    model, tokenizer = loader.load_model(
        model_name=MODEL_NAME,
        quantization="4bit" if USE_QUANTIZATION else None,
    )

    logger.info(f"Model loaded. Parameters before LoRA:")
    loader.print_trainable_parameters(model)

    # 2. Apply LoRA
    logger.info("\n2. Applying LoRA...")

    # Option A: Manual configuration
    lora_config = LoRAConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["c_attn", "c_proj"],  # GPT-2 specific
        lora_dropout=0.1,
    )

    # Option B: Automatic configuration for model type
    # lora_config = create_lora_config_for_model(
    #     model_type="gpt2",
    #     r=LORA_R,
    #     lora_alpha=LORA_ALPHA,
    #     include_attention=True,
    #     include_mlp=False,  # Only attention layers for efficiency
    # )

    model = apply_lora(
        model=model,
        config=lora_config,
        prepare_for_quantization=USE_QUANTIZATION,
    )

    logger.info(f"LoRA applied. Trainable parameters:")
    loader.print_trainable_parameters(model)

    # 3. Load dataset
    logger.info("\n3. Loading dataset...")
    train_dataset = prepare_dataset(
        dataset_name=DATASET_NAME,
        tokenizer=tokenizer,
        split="train[:2000]",  # Use subset for demo
        text_column="text",
        max_length=256,
        subset=DATASET_SUBSET,
    )

    eval_dataset = prepare_dataset(
        dataset_name=DATASET_NAME,
        tokenizer=tokenizer,
        split="validation[:200]",
        text_column="text",
        max_length=256,
        subset=DATASET_SUBSET,
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # 4. Train
    logger.info("\n4. Training model with LoRA...")
    trainer = FineTuner(
        model=model,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        use_wandb=False,
    )

    result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,  # Higher LR for LoRA
        gradient_accumulation_steps=4,
        warmup_steps=50,
        save_steps=200,
        logging_steps=10,
    )

    logger.info("\n" + "=" * 50)
    logger.info("Training completed!")
    logger.info(f"Results: {result}")
    logger.info(f"LoRA adapters saved to: {OUTPUT_DIR}/final_model")
    logger.info("\nNote: Only LoRA adapters are saved, not the full model.")
    logger.info("To use the model, load the base model and apply these adapters.")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
