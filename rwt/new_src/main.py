#!/usr/bin/env python3
"""
Main Training Script for Random Walk Transformer.

This script handles the complete training pipeline:
- Configuration loading and validation
- Dataset preparation and tokenization
- Model and optimizer creation
- Training with automatic checkpoint resume
- Final evaluation and results logging
"""

import os
import sys
import random
import warnings
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast, 
    get_scheduler,
    set_seed
)

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from config import Config
from logger import setup_logging, get_logger
from trainer import create_trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # HuggingFace transformers seed
    
    # Make CUDA deterministic (may reduce performance)
    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = False


def create_model_and_optimizer(config: Config, num_classes: int):
    """
    Create model, optimizer, scheduler, and scaler.
    
    Args:
        config: Training configuration
        num_classes: Number of output classes
        
    Returns:
        Tuple of (model, optimizer, scheduler, scaler)
    """
    logger = get_logger()
    
    # Import model class
    try:
        from model import Classifier
    except ImportError:
        logger.error("❌ Could not import Classifier model. Please ensure model.py exists.")
        sys.exit(1)
    
    logger.info("🧠 Creating model and optimizer...")
    
    # Create model
    model = Classifier(
        d_model=config.model.d_model,
        nhead=config.model.n_heads,
        num_encoder_layers=config.model.n_layers,
        max_len=config.data.max_length,
        dim_feedforward=config.model.ffn_dim,
        dropout=config.model.dropout,
        num_classes=num_classes,
        vocab_size=config.data.vocab_size,
        # Add any additional model parameters from config
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"📊 Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.model.lr,
        weight_decay=config.model.weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Calculate training steps for scheduler
    # Note: This is approximate, actual steps depend on dataset size
    estimated_steps_per_epoch = 1000  # Will be updated by datamodule
    total_steps = estimated_steps_per_epoch * config.model.num_epochs
    warmup_steps = int(config.model.warmup_ratio * total_steps)
    
    # Create scheduler
    scheduler = get_scheduler(
        "cosine_with_restarts",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        scheduler_specific_kwargs={"num_cycles": 1}
    )
    
    # Create gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    logger.success("✅ Model and optimizer created successfully")
    
    return model, optimizer, scheduler, scaler


def prepare_dataset_and_tokenizer(config: Config):
    """
    Load dataset and prepare tokenizer.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (dataset, tokenizer)
    """
    logger = get_logger()
    
    # Load dataset (local or HuggingFace)
    if config.data.data_path:
        logger.info(f"📂 Loading local dataset: {config.data.data_path}")
        try:
            from data.module import load_local_dataset
            dataset = load_local_dataset(config.data.data_path)
            logger.success(f"✅ Local dataset loaded successfully")
        except ImportError:
            logger.error("❌ Could not import load_local_dataset. Please ensure data/module.py exists.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Failed to load local dataset: {e}")
            sys.exit(1)
    else:
        logger.info(f"📚 Loading HuggingFace dataset: {config.data.repo_id}")
        try:
            dataset = load_dataset(config.data.repo_id)
            logger.success(f"✅ HuggingFace dataset loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load HuggingFace dataset: {e}")
            sys.exit(1)
    
    # Log dataset info
    dataset_info = {}
    for split_name, split_data in dataset.items():
        dataset_info[f"{split_name.title()} Split"] = f"{len(split_data):,} samples"
    
    logger.print_config_table(dataset_info, "📊 Dataset Information")
    
    # Create tokenizer
    try:
        from data.module import create_tokenizer
        tokenizer = create_tokenizer(config, dataset)
    except ImportError:
        logger.warning("⚠️ Could not import create_tokenizer, using default")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    logger.success("✅ Tokenizer ready")
    return dataset, tokenizer


def create_datamodule(config: Config, dataset, tokenizer):
    """
    Create data module for training.
    
    Args:
        config: Training configuration
        dataset: HuggingFace dataset
        tokenizer: Tokenizer
        
    Returns:
        DataModule instance
    """
    logger = get_logger()
    
    try:
        from data.module import DataModule
    except ImportError:
        logger.error("❌ Could not import DataModule. Please ensure data/module.py exists.")
        sys.exit(1)
    
    logger.info("📦 Creating data module...")
    
    datamodule = DataModule(
        dataset=dataset,
        tokenizer=tokenizer,
        config=config
    )
    
    # Log data module info
    data_info = {
        "Train Samples": len(datamodule.train_dataset) if hasattr(datamodule, 'train_dataset') else "Unknown",
        "Val Samples": len(datamodule.val_dataset) if hasattr(datamodule, 'val_dataset') else "Unknown", 
        "Test Samples": len(datamodule.test_dataset) if hasattr(datamodule, 'test_dataset') else "Unknown",
        "Batch Size": config.model.batch_size,
        "Max Length": config.data.max_length,
    }
    
    logger.print_config_table(data_info, "📊 Data Module Information")
    logger.success("✅ Data module created")
    
    return datamodule


def main():
    """Main training function."""
    
    # Parse configuration
    config = Config.from_args()
    
    # Setup logging
    logger = setup_logging(
        log_dir=config.system.logs_dir,
        enable_file_logging=True
    )
    
    # Print startup banner
    logger.print_banner(
        "🚀 Random Walk Transformer Training",
        f"Experiment: {config.system.experiment_name}"
    )
    
    # Set random seeds for reproducibility
    set_seeds(config.system.seed)
    logger.info(f"🎲 Random seed set to: {config.system.seed}")
    
    # Log system and configuration info
    logger.print_system_info()
    
    # Validate configuration
    try:
        # Add any config validation here
        logger.info("✅ Configuration validated")
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        sys.exit(1)
    
    # Prepare dataset and tokenizer
    dataset, tokenizer = prepare_dataset_and_tokenizer(config)
    
    # Get number of classes from dataset
    num_classes = dataset["train"].features["label"].num_classes
    logger.info(f"🎯 Number of classes: {num_classes}")
    
    # Create data module
    datamodule = create_datamodule(config, dataset, tokenizer)
    
    # Create model, optimizer, scheduler, and scaler
    model, optimizer, scheduler, scaler = create_model_and_optimizer(config, num_classes)
    
    # Create trainer (automatically selects PlainTrainer or ContrastiveTrainer)
    trainer = create_trainer(model, optimizer, scheduler, scaler, config)
    
    # Log final setup info
    trainer_type = "Contrastive" if config.model.lambda_ > 0 else "Plain"
    setup_info = {
        "Trainer Type": trainer_type,
        "Device": config.system.device,
        "Mixed Precision": "Enabled",
        "Checkpoint Dir": str(config.get_checkpoint_dir()),
        "Resume": "Enabled" if trainer._has_checkpoint() else "Disabled"
    }
    logger.print_config_table(setup_info, "🎯 Training Setup")
    
    try:
        # Start training
        logger.print_banner("🏁 Starting Training Loop")
        test_accuracy = trainer.train(datamodule)
        
        # Training completed successfully
        logger.print_banner(
            "🎉 Training Completed Successfully!",
            f"Final Test Accuracy: {test_accuracy:.4f}"
        )
        
        # Log final summary
        final_summary = {
            "Final Test Accuracy": f"{test_accuracy:.4f}",
            "Best Validation Loss": f"{trainer.best_val_loss:.4f}",
            "Total Training Steps": trainer.step,
            "Experiment Name": config.system.experiment_name,
            "Checkpoint Location": str(trainer.checkpoint_dir)
        }
        logger.print_metrics_table(final_summary, "🏆 Final Results")
        
        return test_accuracy
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user")
        logger.info("💾 Checkpoint saved, you can resume training later")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.log_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()