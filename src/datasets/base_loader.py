"""Base dataset loaders for general LLM fine-tuning"""

from typing import Optional, Dict, Any, List, Union
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDatasetLoader:
    """
    Loader for text datasets.

    Supports HuggingFace datasets and custom data.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize dataset loader.

        Args:
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_huggingface_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        subset: Optional[str] = None,
        streaming: bool = False,
    ) -> Dataset:
        """
        Load dataset from HuggingFace.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split to load
            text_column: Name of the text column
            subset: Dataset subset/config
            streaming: Whether to stream the dataset

        Returns:
            Dataset object
        """
        logger.info(f"Loading dataset: {dataset_name}")

        if subset:
            dataset = load_dataset(dataset_name, subset, split=split, streaming=streaming)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)

        logger.info(f"Dataset loaded with {len(dataset) if not streaming else 'streaming'} examples")

        return dataset

    def load_from_text_file(
        self,
        file_path: str,
        delimiter: str = "\n\n",
    ) -> Dataset:
        """
        Load dataset from text file.

        Args:
            file_path: Path to text file
            delimiter: Delimiter between examples

        Returns:
            Dataset object
        """
        logger.info(f"Loading text from: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split by delimiter
        examples = text.split(delimiter)
        examples = [ex.strip() for ex in examples if ex.strip()]

        dataset = Dataset.from_dict({"text": examples})

        logger.info(f"Loaded {len(dataset)} examples from file")

        return dataset

    def tokenize_function(
        self,
        examples: Dict[str, List],
        text_column: str = "text",
    ) -> Dict[str, List]:
        """
        Tokenize examples.

        Args:
            examples: Dictionary with text examples
            text_column: Name of text column

        Returns:
            Dictionary with tokenized examples
        """
        return self.tokenizer(
            examples[text_column],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )

    def prepare_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        num_proc: int = 4,
        remove_columns: Optional[List[str]] = None,
    ) -> Dataset:
        """
        Tokenize and prepare dataset for training.

        Args:
            dataset: Input dataset
            text_column: Name of text column
            num_proc: Number of processes for tokenization
            remove_columns: Columns to remove after tokenization

        Returns:
            Prepared dataset
        """
        logger.info("Tokenizing dataset...")

        if remove_columns is None:
            remove_columns = dataset.column_names

        tokenized_dataset = dataset.map(
            lambda x: self.tokenize_function(x, text_column),
            batched=True,
            num_proc=num_proc,
            remove_columns=remove_columns,
            desc="Tokenizing",
        )

        logger.info("Dataset prepared for training")

        return tokenized_dataset


def prepare_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    text_column: str = "text",
    max_length: int = 512,
    subset: Optional[str] = None,
) -> Dataset:
    """
    Convenience function to load and prepare a dataset.

    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer to use
        split: Dataset split
        text_column: Text column name
        max_length: Maximum sequence length
        subset: Dataset subset

    Returns:
        Prepared dataset
    """
    loader = TextDatasetLoader(tokenizer, max_length)
    dataset = loader.load_huggingface_dataset(
        dataset_name,
        split=split,
        text_column=text_column,
        subset=subset,
    )
    prepared = loader.prepare_dataset(dataset, text_column=text_column)
    return prepared


class InstructionDatasetLoader(TextDatasetLoader):
    """
    Loader for instruction-tuning datasets.

    Formats instruction + input + output into a single sequence.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
    ):
        """
        Initialize instruction dataset loader.

        Args:
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            instruction_template: Template for formatting instructions
        """
        super().__init__(tokenizer, max_length)
        self.instruction_template = instruction_template

    def format_instruction(
        self,
        instruction: str,
        input_text: str = "",
        output: str = "",
    ) -> str:
        """
        Format instruction, input, and output into a single string.

        Args:
            instruction: Instruction text
            input_text: Input text
            output: Expected output

        Returns:
            Formatted string
        """
        return self.instruction_template.format(
            instruction=instruction,
            input=input_text,
            output=output,
        )

    def prepare_instruction_dataset(
        self,
        dataset: Dataset,
        instruction_column: str = "instruction",
        input_column: str = "input",
        output_column: str = "output",
    ) -> Dataset:
        """
        Prepare instruction dataset.

        Args:
            dataset: Input dataset
            instruction_column: Instruction column name
            input_column: Input column name
            output_column: Output column name

        Returns:
            Prepared dataset
        """
        logger.info("Formatting instruction dataset...")

        def format_fn(examples):
            texts = []
            for inst, inp, out in zip(
                examples[instruction_column],
                examples[input_column],
                examples[output_column],
            ):
                text = self.format_instruction(inst, inp, out)
                texts.append(text)
            return {"text": texts}

        formatted_dataset = dataset.map(
            format_fn,
            batched=True,
            desc="Formatting instructions",
        )

        # Tokenize
        prepared = self.prepare_dataset(
            formatted_dataset,
            text_column="text",
        )

        return prepared
