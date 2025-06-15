"""
Simple Training Configuration Classes.

Three main configuration classes for clean organization:
1. DataConfig - Dataset and preprocessing
2. ModelConfig - Model architecture and training
3. SystemConfig - System settings and paths
"""

import os
import json
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""
    
    # Dataset
    dataset: str = "pubmed"
    data_path: str = ""  # Local dataset path (takes priority over repo_id)
    repo_id: str = ""  # HuggingFace repo ID (fallback)
    
    # Random walk parameters
    walk_len: int = 7
    group_size: int = 15
    p: float = 1.0
    q: Optional[float] = None
    
    # Preprocessing
    max_length: int = 512
    vocab_size: int = 30522
    tokenizer_path: str = ""  # Custom tokenizer path
    anonymize: bool = False
    split_seed: Optional[int] = None
    
    def __post_init__(self):
        if self.q is None:
            self.q = 1.0 / self.p
        if not self.data_path and not self.repo_id:
            raise ValueError("Either data_path (local) or repo_id (HuggingFace) is required")
        
        # Auto-extract dataset name from data_path if not provided
        if self.data_path and self.dataset == "pubmed":  # "pubmed" is default
            self.dataset = self._extract_dataset_name_from_path(self.data_path)
    
    def _extract_dataset_name_from_path(self, data_path: str) -> str:
        """Extract dataset name from data path."""
        from pathlib import Path
        
        path = Path(data_path)
        
        # Try to extract from directory name
        # Common patterns: "data/cora_standard", "datasets/pubmed_node2vec", etc.
        dir_name = path.name.lower()
        
        # Known dataset names to look for
        known_datasets = ["cora", "pubmed", "citeseer", "amazon", "roman"]
        
        for dataset in known_datasets:
            if dataset in dir_name:
                return dataset
        
        # If no known dataset found, use the directory name itself
        # Remove common suffixes like "_standard", "_node2vec", etc.
        clean_name = dir_name
        suffixes_to_remove = ["_standard", "_node2vec", "_nbw", "_mdlr", "_anon", "_masked"]
        for suffix in suffixes_to_remove:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        return clean_name


@dataclass 
class ModelConfig:
    """Model architecture and training configuration."""
    
    # Model architecture
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 8
    ffn_dim: int = 512
    dropout: float = 0.1
    
    # Training hyperparameters
    lr: float = 1e-5
    batch_size: int = 128
    num_epochs: int = 50
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Advanced training
    grad_acc: int = 1
    lambda_: float = 0.0  # Contrastive loss weight
    max_neg: int = 5
    
    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_acc


@dataclass
class SystemConfig:
    """System settings and paths."""
    
    # Hardware
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu: bool = field(default_factory=lambda: torch.cuda.device_count() > 1)
    seed: int = 42
    
    # Paths
    data_dir: str = "data"
    logs_dir: str = "logs" 
    results_dir: str = "results"
    
    # Logging
    wandb_project: str = "rw-transformer"
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        # Create directories
        for dir_path in [self.data_dir, self.logs_dir, self.results_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Auto-generate experiment name
        if self.experiment_name is None:
            from datetime import datetime
            self.experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory for this specific configuration."""
        import hashlib
        
        # Create a hash from important config values that affect training
        config_str = (
            f"dataset_{self.data.dataset}_"
            f"repo_{self.data.repo_id.replace('/', '_')}_"
            f"model_{self.model.d_model}_{self.model.n_layers}_{self.model.n_heads}_"
            f"train_{self.model.lr}_{self.model.batch_size}_{self.model.num_epochs}_"
            f"data_{self.data.walk_len}_{self.data.group_size}_{self.data.p}_{self.data.q}_"
            f"lambda_{self.model.lambda_}"
        )
        
        # Create short hash for uniqueness
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create directory name: experiment_name + config_hash
        if self.experiment_name:
            dir_name = f"{self.experiment_name}_{config_hash}"
        else:
            dir_name = f"config_{config_hash}"
        
        ckpt_dir = Path(self.results_dir) / dir_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir


@dataclass
class Config:
    """Main configuration class combining all settings."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig) 
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def from_args(cls) -> 'Config':
        """Create config from command line arguments."""
        parser = argparse.ArgumentParser(description="Random Walk Transformer Training")
        
        # Data arguments
        data_group = parser.add_argument_group('Data')
        data_group.add_argument("--dataset", default="pubmed", help="Dataset name")
        data_group.add_argument("--data_path", help="Local dataset path (Arrow/Parquet format)")
        data_group.add_argument("--repo_id", help="HuggingFace repo ID (fallback if no data_path)")
        data_group.add_argument("--walk_len", type=int, default=7, help="Random walk length")
        data_group.add_argument("--group_size", type=int, default=15, help="Group size")
        data_group.add_argument("--p", type=float, default=1.0, help="Return parameter")
        data_group.add_argument("--q", type=float, help="In-out parameter")
        data_group.add_argument("--max_length", type=int, default=512, help="Max sequence length")
        data_group.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size")
        data_group.add_argument("--tokenizer_path", help="Custom tokenizer path")
        data_group.add_argument("--anonymize", action="store_true", help="Anonymize dataset")
        data_group.add_argument("--split_seed", type=int, help="Data split seed")
        
        # Model arguments  
        model_group = parser.add_argument_group('Model')
        model_group.add_argument("--d_model", type=int, default=128, help="Model dimension")
        model_group.add_argument("--n_layers", type=int, default=6, help="Number of layers")
        model_group.add_argument("--n_heads", type=int, default=8, help="Number of heads")
        model_group.add_argument("--ffn_dim", type=int, default=512, help="FFN dimension")
        model_group.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
        model_group.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
        model_group.add_argument("--batch_size", type=int, default=128, help="Batch size")
        model_group.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
        model_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
        model_group.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
        model_group.add_argument("--grad_acc", type=int, default=1, help="Gradient accumulation")
        model_group.add_argument("--lambda", type=float, default=0.0, dest='lambda_', 
                                help="Contrastive loss weight")
        model_group.add_argument("--max_neg", type=int, default=5, help="Max negative samples")
        
        # System arguments
        system_group = parser.add_argument_group('System')
        system_group.add_argument("--seed", type=int, default=42, help="Random seed")
        system_group.add_argument("--data_dir", default="data", help="Data directory")
        system_group.add_argument("--logs_dir", default="logs", help="Logs directory") 
        system_group.add_argument("--results_dir", default="results", help="Results directory")
        system_group.add_argument("--wandb_project", default="rw-transformer", help="W&B project")
        system_group.add_argument("--experiment_name", help="Experiment name")
        
        args = parser.parse_args()
        args_dict = vars(args)
        
        # Create configs from args
        data_config = DataConfig(
            dataset=args_dict['dataset'],
            data_path=args_dict.get('data_path', ''),
            repo_id=args_dict.get('repo_id', ''),
            walk_len=args_dict['walk_len'],
            group_size=args_dict['group_size'],
            p=args_dict['p'],
            q=args_dict['q'],
            max_length=args_dict['max_length'],
            vocab_size=args_dict['vocab_size'],
            tokenizer_path=args_dict.get('tokenizer_path', ''),
            anonymize=args_dict['anonymize'],
            split_seed=args_dict['split_seed']
        )
        
        model_config = ModelConfig(
            d_model=args_dict['d_model'],
            n_layers=args_dict['n_layers'],
            n_heads=args_dict['n_heads'],
            ffn_dim=args_dict['ffn_dim'],
            dropout=args_dict['dropout'],
            lr=args_dict['lr'],
            batch_size=args_dict['batch_size'],
            num_epochs=args_dict['num_epochs'],
            weight_decay=args_dict['weight_decay'],
            warmup_ratio=args_dict['warmup_ratio'],
            grad_acc=args_dict['grad_acc'],
            lambda_=args_dict['lambda_'],
            max_neg=args_dict['max_neg']
        )
        
        system_config = SystemConfig(
            seed=args_dict['seed'],
            data_dir=args_dict['data_dir'],
            logs_dir=args_dict['logs_dir'],
            results_dir=args_dict['results_dir'],
            wandb_project=args_dict['wandb_project'],
            experiment_name=args_dict['experiment_name']
        )
        
        return cls(data=data_config, model=model_config, system=system_config)
    
    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory for this specific configuration."""
        import hashlib
        
        # Create a hash from important config values that affect training
        config_str = (
            f"dataset_{self.data.dataset}_"
            f"repo_{self.data.repo_id.replace('/', '_') if self.data.repo_id else 'local'}_"
            f"path_{self.data.data_path.replace('/', '_') if self.data.data_path else 'none'}_"
            f"model_{self.model.d_model}_{self.model.n_layers}_{self.model.n_heads}_"
            f"train_{self.model.lr}_{self.model.batch_size}_{self.model.num_epochs}_"
            f"data_{self.data.walk_len}_{self.data.group_size}_{self.data.p}_{self.data.q}_"
            f"lambda_{self.model.lambda_}"
        )
        
        # Create short hash for uniqueness
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create directory name: experiment_name + config_hash
        if self.system.experiment_name:
            dir_name = f"{self.system.experiment_name}_{config_hash}"
        else:
            dir_name = f"config_{config_hash}"
        
        ckpt_dir = Path(self.system.results_dir) / dir_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir
    
    def save(self, path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            data=DataConfig(**data['data']),
            model=ModelConfig(**data['model']),
            system=SystemConfig(**data['system'])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def __repr__(self) -> str:
        return (f"Config(\n"
                f"  Dataset: {self.data.dataset} (repo: {self.data.repo_id})\n"
                f"  Model: {self.model.d_model}d, {self.model.n_layers}L, {self.model.n_heads}H\n"
                f"  Training: lr={self.model.lr}, bs={self.model.batch_size}, epochs={self.model.num_epochs}\n"
                f"  Device: {self.system.device}\n"
                f")")


# Convenience function
def load_config() -> Config:
    """Load config from command line arguments."""
    return Config.from_args()