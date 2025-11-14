"""
Basic fine-tuning example without LoRA.

This example demonstrates standard full fine-tuning of a small LLM.
"""

import sys
sys.path.append("..")

from src.core.model_loader import ModelLoader
from src.core.trainer import FineTuner
from src.datasets.base_loader import prepare_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run basic fine-tuning example."""

    logger.info("=" * 50)
    logger.info("Basic Fine-Tuning Example")
    logger.info("=" * 50)

    # Configuration
    MODEL_NAME = "gpt2"
    DATASET_NAME = "wikitext"
    DATASET_SUBSET = "wikitext-2-raw-v1"
    OUTPUT_DIR = "./outputs/basic_finetuning"

    # 1. Load model and tokenizer
    logger.info("\n1. Loading model...")
    loader = ModelLoader()
    model, tokenizer = loader.load_model(
        model_name=MODEL_NAME,
        quantization=None,  # No quantization for basic fine-tuning
        torch_dtype="float32",
    )

    loader.print_trainable_parameters(model)

    # 2. Load and prepare dataset
    logger.info("\n2. Loading dataset...")
    train_dataset = prepare_dataset(
        dataset_name=DATASET_NAME,
        tokenizer=tokenizer,
        split="train[:1000]",  # Use small subset for demo
        text_column="text",
        max_length=256,
        subset=DATASET_SUBSET,
    )

    eval_dataset = prepare_dataset(
        dataset_name=DATASET_NAME,
        tokenizer=tokenizer,
        split="validation[:100]",
        text_column="text",
        max_length=256,
        subset=DATASET_SUBSET,
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # 3. Create trainer and train
    logger.info("\n3. Training model...")
    trainer = FineTuner(
        model=model,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        use_wandb=False,
    )

    # Train with custom arguments
    result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=1,
        batch_size=2,
        learning_rate=5e-5,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=10,
    )

    logger.info("\n" + "=" * 50)
    logger.info("Training completed!")
    logger.info(f"Results: {result}")
    logger.info(f"Model saved to: {OUTPUT_DIR}/final_model")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
