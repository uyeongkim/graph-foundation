"""
Data Module for Random Walk Transformer Training.

Handles local dataset loading (Arrow/SafeTensor format), preprocessing, tokenization, and DataLoader creation.
"""

import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset, DatasetDict, load_from_disk
from transformers import PreTrainedTokenizerFast

from logger import get_logger
from config import Config

logger = get_logger(__name__)


class RandomWalkDataset(Dataset):
    """
    Dataset wrapper for random walk text data.
    
    Handles tokenization and formatting for PyTorch DataLoader.
    """
    
    def __init__(
        self,
        data: HFDataset,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        text_column: str = "string_rw",
        label_column: str = "label", 
        node_idx_column: str = "node_idx"
    ):
        """
        Initialize dataset.
        
        Args:
            data: HuggingFace dataset
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            text_column: Name of text column in dataset
            label_column: Name of label column in dataset
            node_idx_column: Name of node index column
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.node_idx_column = node_idx_column
        
        # Validate columns exist
        self._validate_columns()
        
        logger.info(f"📊 Dataset created with {len(self.data)} samples")
    
    def _validate_columns(self):
        """Validate that required columns exist in dataset."""
        required_columns = [self.text_column, self.label_column, self.node_idx_column]
        available_columns = list(self.data.column_names)
        
        missing_columns = [col for col in required_columns if col not in available_columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {available_columns}"
            )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - input_ids: Tokenized text
            - attention_mask: Attention mask
            - labels: Ground truth label
            - node_idx: Node index for majority voting
        """
        item = self.data[idx]
        
        # Get text and tokenize
        text = item[self.text_column]
        if isinstance(text, list):
            # If text is a list, join with spaces
            text = " ".join(str(t) for t in text)
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Prepare output
        output = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item[self.label_column], dtype=torch.long),
            "node_idx": torch.tensor(item[self.node_idx_column], dtype=torch.long)
        }
        
        return output


class DataModule:
    """
    Data module for handling all data-related operations.
    
    Features:
    - Dataset loading and preprocessing
    - Train/validation/test split
    - DataLoader creation with proper settings
    - Tokenizer management
    """
    
    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: PreTrainedTokenizerFast,
        config: Config
    ):
        """
        Initialize data module.
        
        Args:
            dataset: HuggingFace dataset with train/val/test splits
            tokenizer: Tokenizer for text processing
            config: Training configuration
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config
        
        # Dataset configuration
        self.max_length = config.data.max_length
        self.batch_size = config.model.batch_size
        
        # Create datasets
        self._create_datasets()
        
        # Log dataset information
        self._log_dataset_info()
    
    def _create_datasets(self):
        """Create train, validation, and test datasets."""
        logger.info("📦 Creating datasets...")
        
        # Create wrapped datasets
        self.train_dataset = RandomWalkDataset(
            data=self.dataset["train"],
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        self.val_dataset = RandomWalkDataset(
            data=self.dataset["validation"],
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        self.test_dataset = RandomWalkDataset(
            data=self.dataset["test"],
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        logger.success("✅ Datasets created successfully")
    
    def _log_dataset_info(self):
        """Log dataset information."""
        dataset_info = {
            "Train Samples": len(self.train_dataset),
            "Validation Samples": len(self.val_dataset),
            "Test Samples": len(self.test_dataset),
            "Max Length": self.max_length,
            "Batch Size": self.batch_size,
            "Vocab Size": len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else "Unknown"
        }
        
        # Get label distribution for train set
        if hasattr(self.dataset["train"], "features") and "label" in self.dataset["train"].features:
            num_classes = self.dataset["train"].features["label"].num_classes
            dataset_info["Number of Classes"] = num_classes
        
        logger.print_config_table(dataset_info, "📊 Dataset Information")
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._get_num_workers(),
            pin_memory=True,
            drop_last=True,  # For stable batch sizes during training
            persistent_workers=True if self._get_num_workers() > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self._get_num_workers(),
            pin_memory=True,
            persistent_workers=True if self._get_num_workers() > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self._get_num_workers(),
            pin_memory=True,
            persistent_workers=True if self._get_num_workers() > 0 else False
        )
    
    def _get_num_workers(self) -> int:
        """
        Get optimal number of workers for DataLoader.
        
        Returns:
            Number of workers (0 for single process, >0 for multiprocessing)
        """
        # Use 4 workers by default, but can be adjusted based on system
        num_workers = min(4, os.cpu_count() or 1)
        
        # Reduce workers if running on single GPU or small dataset
        if len(self.train_dataset) < 1000:
            num_workers = min(2, num_workers)
        
        return num_workers
    
    def get_sample_batch(self, split: str = "train") -> Dict[str, torch.Tensor]:
        """
        Get a sample batch for debugging/testing.
        
        Args:
            split: Which split to sample from ('train', 'val', 'test')
            
        Returns:
            Sample batch dictionary
        """
        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Get first item
        sample = dataset[0]
        
        # Add batch dimension
        batch = {}
        for key, value in sample.items():
            batch[key] = value.unsqueeze(0)
        
        return batch
    
    def analyze_sequence_lengths(self) -> Dict[str, Any]:
        """
        Analyze sequence lengths in the dataset.
        
        Returns:
            Dictionary with length statistics
        """
        logger.info("📏 Analyzing sequence lengths...")
        
        lengths = []
        
        # Sample some texts to analyze
        sample_size = min(1000, len(self.train_dataset))
        indices = torch.randperm(len(self.train_dataset))[:sample_size]
        
        for idx in indices:
            item = self.train_dataset[idx]
            # Count non-padding tokens
            length = (item["attention_mask"] == 1).sum().item()
            lengths.append(length)
        
        lengths = torch.tensor(lengths)
        
        stats = {
            "Mean Length": lengths.float().mean().item(),
            "Median Length": lengths.float().median().item(),
            "Min Length": lengths.min().item(),
            "Max Length": lengths.max().item(),
            "Std Length": lengths.float().std().item(),
            "95th Percentile": torch.quantile(lengths.float(), 0.95).item(),
            "99th Percentile": torch.quantile(lengths.float(), 0.99).item(),
        }
        
        logger.print_metrics_table(stats, "📏 Sequence Length Analysis")
        return stats
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get class distribution in training set.
        
        Returns:
            Dictionary mapping class indices to counts
        """
        logger.info("📊 Analyzing class distribution...")
        
        labels = []
        for item in self.train_dataset:
            labels.append(item["labels"].item())
        
        # Count classes
        class_counts = {}
        for label in set(labels):
            class_counts[f"Class {label}"] = labels.count(label)
        
        logger.print_metrics_table(class_counts, "📊 Class Distribution")
        return class_counts


def load_local_dataset(data_path: Union[str, Path]) -> DatasetDict:
    """
    Load dataset from local Arrow/SafeTensor files.
    
    Args:
        data_path: Path to dataset directory or specific dataset file
        
    Returns:
        DatasetDict with train/validation/test splits
    """
    logger.info(f"📂 Loading local dataset from: {data_path}")
    
    data_path = Path(data_path)
    
    try:
        if data_path.is_dir():
            # Load from directory (Arrow format)
            dataset = load_from_disk(str(data_path))
            logger.success(f"✅ Loaded dataset from directory: {data_path}")
            
        elif data_path.suffix in ['.arrow', '.dataset']:
            # Load single Arrow file
            dataset = load_from_disk(str(data_path))
            logger.success(f"✅ Loaded dataset from Arrow file: {data_path}")
            
        else:
            # Try to load as a directory anyway
            dataset = load_from_disk(str(data_path))
            logger.success(f"✅ Loaded dataset: {data_path}")
            
    except Exception as e:
        logger.error(f"❌ Failed to load dataset from {data_path}: {e}")
        raise
    
    # Validate dataset structure
    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"Expected DatasetDict, got {type(dataset)}")
    
    required_splits = ['train', 'validation', 'test']
    available_splits = list(dataset.keys())
    
    # Check if we have required splits
    missing_splits = [split for split in required_splits if split not in available_splits]
    if missing_splits:
        logger.warning(f"⚠️ Missing splits: {missing_splits}. Available: {available_splits}")
        
        # Try common alternative names
        if 'val' in available_splits and 'validation' not in available_splits:
            dataset['validation'] = dataset['val']
            logger.info("📝 Mapped 'val' to 'validation'")
        
        # If still missing validation, create from train
        if 'validation' not in dataset and 'train' in dataset:
            logger.info("🔄 Creating validation split from training data")
            train_test = dataset['train'].train_test_split(test_size=0.1, seed=42)
            dataset['train'] = train_test['train']
            dataset['validation'] = train_test['test']
    
    # Log dataset info
    dataset_info = {}
    for split_name, split_data in dataset.items():
        dataset_info[f"{split_name.title()} Split"] = f"{len(split_data):,} samples"
    
    logger.print_config_table(dataset_info, "📊 Local Dataset Information")
    
    return dataset


def auto_detect_dataset_format(data_path: Union[str, Path]) -> DatasetDict:
    """
    Auto-detect and load dataset from various local formats.
    
    Args:
        data_path: Path to dataset
        
    Returns:
        DatasetDict
    """
    data_path = Path(data_path)
    
    # Try different loading strategies
    strategies = [
        ("Arrow/Parquet Directory", lambda: load_from_disk(str(data_path))),
        ("Arrow File", lambda: load_from_disk(str(data_path)) if data_path.suffix == '.arrow' else None),
    ]
    
    for strategy_name, loader_func in strategies:
        try:
            logger.info(f"🔍 Trying {strategy_name}...")
            result = loader_func()
            if result is not None:
                logger.success(f"✅ Successfully loaded using {strategy_name}")
                return result
        except Exception as e:
            logger.debug(f"❌ {strategy_name} failed: {e}")
            continue
    
    raise ValueError(f"Could not load dataset from {data_path} using any supported format")


def create_tokenizer(config: Config, dataset: Optional[DatasetDict] = None) -> PreTrainedTokenizerFast:
    """
    Create or load tokenizer.
    
    Args:
        config: Training configuration
        dataset: Dataset for training tokenizer (if needed)
        
    Returns:
        PreTrainedTokenizerFast tokenizer
    """
    logger.info("🔧 Setting up tokenizer...")
    
    # Check if custom tokenizer path is specified
    tokenizer_path = None
    if hasattr(config.data, 'tokenizer_path') and config.data.tokenizer_path:
        tokenizer_path = Path(config.data.tokenizer_path)
    else:
        # Default tokenizer path
        tokenizer_path = Path("data/tokenizer") / f"vocab_{config.data.vocab_size}"
    
    if tokenizer_path.exists():
        logger.info(f"📖 Loading existing tokenizer from {tokenizer_path}")
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
        except Exception as e:
            logger.warning(f"⚠️ Failed to load custom tokenizer: {e}")
            logger.info("🔄 Falling back to default tokenizer")
            tokenizer = _create_default_tokenizer()
    else:
        logger.info("🆕 Using default tokenizer...")
        tokenizer = _create_default_tokenizer()
        
        # Optionally save for future use
        if tokenizer_path.parent.exists() or True:  # Create directory
            try:
                tokenizer_path.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(str(tokenizer_path))
                logger.info(f"💾 Default tokenizer saved to {tokenizer_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not save tokenizer: {e}")
    
    # Ensure special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    
    logger.success(f"✅ Tokenizer ready (vocab size: {len(tokenizer)})")
    return tokenizer


def _create_default_tokenizer() -> PreTrainedTokenizerFast:
    """Create a default tokenizer."""
    from transformers import AutoTokenizer
    
    # Use a fast, reliable tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    return tokenizer