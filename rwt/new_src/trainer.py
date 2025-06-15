"""
Enhanced Trainer with Resume Functionality and Rich Logging.

Single GPU trainer with automatic checkpoint resume, early stopping,
and beautiful terminal output using Rich.
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.nn.utils
import wandb
from tqdm.rich import tqdm

from logger import get_logger, log_training_start, log_training_complete
from config import Config


class BaseTrainer:
    """
    Base trainer class with checkpoint resume functionality.
    
    Features:
    - Automatic checkpoint saving and resuming
    - Early stopping with patience
    - Rich logging and progress tracking
    - Weights & Biases integration
    - Mixed precision training
    """
    
    def __init__(
        self, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler: torch.cuda.amp.GradScaler,
        config: Config
    ):
        """
        Initialize trainer with model components and configuration.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            scaler: AMP gradient scaler
            config: Training configuration
        """
        # Store components
        self.config = config
        self.device = config.system.device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        
        # Setup logging
        self.logger = get_logger()
        
        # Initialize training state
        self._initialize_training_state()
        
        # Setup checkpoint directory
        self.checkpoint_dir = config.get_checkpoint_dir()
        
        # Check for existing checkpoint and resume if available
        if self._has_checkpoint():
            self._resume_from_checkpoint()
        else:
            self.logger.info("🆕 Starting fresh training")
            self._log_initial_setup()
        
        # Setup W&B logging
        self._setup_wandb()
    
    def _initialize_training_state(self):
        """Initialize training state variables."""
        self.start_epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = max(1, int(0.3 * self.config.model.num_epochs))
        
    def _log_initial_setup(self):
        """Log initial training setup information."""
        # Log training start with config
        config_dict = self.config.to_dict()
        log_training_start(config_dict, self.model)
        
        # Log training parameters
        training_info = {
            "Total Epochs": self.config.model.num_epochs,
            "Batch Size": self.config.model.batch_size,
            "Effective Batch Size": self.config.model.effective_batch_size,
            "Learning Rate": self.config.model.lr,
            "Device": self.device,
            "Patience": self.patience,
            "Checkpoint Dir": str(self.checkpoint_dir)
        }
        self.logger.print_config_table(training_info, "🚂 Training Parameters")
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if not wandb.run:
            wandb.init(
                project=self.config.system.wandb_project,
                name=self.config.system.experiment_name,
                config=self.config.to_dict(),
                resume="allow",
                id=self.config.system.experiment_name
            )
            self.logger.info("📊 Weights & Biases initialized")
    
    def _has_checkpoint(self) -> bool:
        """Check if checkpoint files exist."""
        required_files = [
            "best_model.pt",
            "optimizer.pt",
            "training_state.json"
        ]
        return all((self.checkpoint_dir / f).exists() for f in required_files)
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Save checkpoint with current training state.
        
        Args:
            epoch: Current epoch
            val_loss: Current validation loss
            is_best: Whether this is the best model so far
        """
        # Save model state (always save current, and best if is_best)
        torch.save(self.model.state_dict(), self.checkpoint_dir / "current_model.pt")
        if is_best:
            torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model.pt")
            self.logger.success(f"💾 Best model saved at epoch {epoch}")
        
        # Save optimizer and scheduler state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict()
        }, self.checkpoint_dir / "optimizer.pt")
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'step': self.step,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'completed': False
        }
        with open(self.checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        # Save configuration
        self.config.save(self.checkpoint_dir / "config.json")
    
    def _resume_from_checkpoint(self):
        """Resume training from checkpoint."""
        self.logger.info("📂 Resuming from checkpoint...")
        
        # Load model state (use best model for resume)
        if (self.checkpoint_dir / "best_model.pt").exists():
            model_path = self.checkpoint_dir / "best_model.pt"
        else:
            model_path = self.checkpoint_dir / "current_model.pt"
            
        model_state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_state)
        
        # Load optimizer and scheduler state
        opt_checkpoint = torch.load(self.checkpoint_dir / "optimizer.pt", map_location=self.device)
        self.optimizer.load_state_dict(opt_checkpoint['optimizer'])
        self.scheduler.load_state_dict(opt_checkpoint['scheduler'])
        self.scaler.load_state_dict(opt_checkpoint['scaler'])
        
        # Load training state
        with open(self.checkpoint_dir / "training_state.json", 'r') as f:
            training_state = json.load(f)
        
        self.start_epoch = training_state['epoch'] + 1
        self.step = training_state['step']
        self.best_val_loss = training_state['best_val_loss']
        self.patience_counter = training_state['patience_counter']
        
        # Log resume information
        resume_info = {
            "Resumed Epoch": self.start_epoch,
            "Total Steps": self.step,
            "Best Val Loss": f"{self.best_val_loss:.4f}",
            "Patience Counter": f"{self.patience_counter}/{self.patience}",
            "Remaining Epochs": self.config.model.num_epochs - self.start_epoch
        }
        self.logger.print_metrics_table(resume_info, "📂 Resume Information")
        self.logger.success(f"✅ Successfully resumed from epoch {self.start_epoch}")
    
    def _mark_training_complete(self):
        """Mark training as completed in checkpoint."""
        training_state_path = self.checkpoint_dir / "training_state.json"
        if training_state_path.exists():
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
            
            training_state['completed'] = True
            
            with open(training_state_path, 'w') as f:
                json.dump(training_state, f, indent=2)
    
    def train(self, datamodule) -> float:
        """
        Main training loop.
        
        Args:
            datamodule: Data module with train/val/test dataloaders
            
        Returns:
            Final test accuracy
        """
        if self.start_epoch >= self.config.model.num_epochs:
            self.logger.warning("🏁 Training already completed!")
            return self.test(datamodule.test_dataloader())
        
        # Training banner
        self.logger.print_banner(
            "🚂 Training Started", 
            f"Epochs: {self.start_epoch} → {self.config.model.num_epochs}"
        )
        
        # Main training loop
        for epoch in range(self.start_epoch, self.config.model.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loader = datamodule.train_dataloader()
            train_metrics = self._run_epoch(train_loader, training=True, epoch=epoch)
            
            # Validation phase (every 5 epochs or last epoch)
            if epoch % 5 == 0 or epoch == self.config.model.num_epochs - 1:
                self.model.eval()
                val_loader = datamodule.val_dataloader()
                val_loss, val_acc = self.evaluate(val_loader)
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Check for best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.logger.success(f"🎉 New best validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience and epoch >= int(0.1 * self.config.model.num_epochs):
                        self.logger.warning(
                            f"🛑 Early stopping at epoch {epoch} "
                            f"(no improvement for {self.patience} checks)"
                        )
                        break
                
                # Save checkpoint
                self._save_checkpoint(epoch, val_loss, is_best)
                
                # Log epoch summary
                epoch_metrics = {
                    "Epoch": f"{epoch + 1}/{self.config.model.num_epochs}",
                    "Train Loss": f"{train_metrics.get('loss', 0.0):.4f}",
                    "Val Loss": f"{val_loss:.4f}",
                    "Val Accuracy": f"{val_acc:.4f}",
                    "Learning Rate": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                    "Epoch Time": f"{epoch_time:.1f}s",
                    "Patience": f"{self.patience_counter}/{self.patience}"
                }
                self.logger.print_metrics_table(epoch_metrics, f"📊 Epoch {epoch + 1} Summary")
                
                # Log to wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics.get('loss', 0.0),
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "patience_counter": self.patience_counter,
                    "epoch_time": epoch_time
                }, step=self.step)
        
        # Mark training as complete
        self._mark_training_complete()
        
        # Final testing
        self.logger.print_banner("🧪 Final Testing", "Evaluating best model on test set")
        
        # Load best model for testing
        if (self.checkpoint_dir / "best_model.pt").exists():
            best_model_state = torch.load(self.checkpoint_dir / "best_model.pt", map_location=self.device)
            self.model.load_state_dict(best_model_state)
            self.logger.info("📖 Loaded best model for testing")
        
        test_loader = datamodule.test_dataloader()
        test_accuracy = self.test(test_loader)
        
        # Log final results
        final_metrics = {
            "Test Accuracy": test_accuracy,
            "Best Val Loss": self.best_val_loss,
            "Total Steps": self.step,
            "Total Epochs Trained": epoch + 1
        }
        log_training_complete(final_metrics)
        
        # Log to wandb
        wandb.log({
            "test/accuracy": test_accuracy,
            "final/best_val_loss": self.best_val_loss
        }, step=self.step)
        
        return test_accuracy
    
    def _run_epoch(self, loader, training: bool = True, epoch: int = 0) -> Dict[str, float]:
        """
        Run one epoch of training or evaluation.
        
        Args:
            loader: DataLoader
            training: Whether this is training or evaluation
            epoch: Current epoch number
            
        Returns:
            Dictionary of metrics for this epoch
        """
        self.model.train() if training else self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        desc = f"Epoch {epoch + 1} ({'Train' if training else 'Eval'})"
        loader_tqdm = tqdm(loader, desc=desc, leave=False)
        
        for batch_idx, batch in enumerate(loader_tqdm):
            # Move batch to device
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            if training:
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    loss = self.training_step(batch)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                loader_tqdm.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.6f}',
                    'step': self.step
                })
                
                # Log to wandb (every 100 steps)
                if self.step % 100 == 0:
                    wandb.log({
                        "train/step_loss": loss.item(),
                        "train/learning_rate": current_lr,
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    }, step=self.step)
                
                self.step += 1
                
            else:
                with torch.no_grad():
                    loss = self.training_step(batch)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}
    
    def evaluate(self, loader) -> Tuple[float, float]:
        """
        Evaluate model on validation/test set.
        
        Args:
            loader: DataLoader for evaluation
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    logits = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )["logits"]
                
                # Calculate loss and accuracy
                loss = F.cross_entropy(logits, batch["labels"])
                total_loss += loss.item()
                
                predictions = logits.argmax(dim=-1)
                correct += (predictions == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def test(self, loader) -> float:
        """
        Test model with majority voting using efficient tensor operations.
        
        Args:
            loader: Test DataLoader
            
        Returns:
            Test accuracy with majority voting
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_node_ids = []
        
        # Collect all predictions
        with torch.no_grad():
            for batch in tqdm(loader, desc="Testing", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    logits = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )["logits"]
                
                predictions = logits.argmax(dim=-1)
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_labels.append(batch["labels"].cpu())
                all_node_ids.append(batch["node_idx"].cpu())
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)  # [total_samples]
        all_labels = torch.cat(all_labels, dim=0)           # [total_samples]
        all_node_ids = torch.cat(all_node_ids, dim=0)       # [total_samples]
        
        # Efficient majority voting using tensor operations
        unique_nodes, inverse_indices = torch.unique(all_node_ids, return_inverse=True)
        num_classes = max(all_predictions.max().item(), all_labels.max().item()) + 1
        
        # Vectorized majority voting
        correct_nodes = 0
        
        for i, node_id in enumerate(unique_nodes):
            # Get mask for current node
            node_mask = (inverse_indices == i)
            
            # Get predictions and labels for this node
            node_preds = all_predictions[node_mask]
            node_labels = all_labels[node_mask]
            
            # Fast majority vote using bincount
            pred_majority = torch.bincount(node_preds, minlength=num_classes).argmax()
            label_majority = torch.bincount(node_labels, minlength=num_classes).argmax()
            
            if pred_majority == label_majority:
                correct_nodes += 1
        
        accuracy = correct_nodes / len(unique_nodes) if len(unique_nodes) > 0 else 0.0
        self.logger.info(f"🎯 Test accuracy (majority voting): {accuracy:.4f} ({correct_nodes}/{len(unique_nodes)})")
        
        return accuracy
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform one training step. To be implemented by subclasses.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss tensor
        """
        raise NotImplementedError("Subclasses must implement training_step")


class PlainTrainer(BaseTrainer):
    """Trainer for standard classification without contrastive learning."""
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Standard cross-entropy training step."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        logits = outputs["logits"]
        loss = F.cross_entropy(logits, batch["labels"])
        return loss


class ContrastiveTrainer(BaseTrainer):
    """Trainer with contrastive learning component."""
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Training step with contrastive loss."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_embeddings=True
        )
        
        # Classification loss
        ce_loss = F.cross_entropy(outputs["logits"], batch["labels"])
        
        # Contrastive loss
        if self.config.model.lambda_ > 0:
            cont_loss = self._contrastive_loss(
                outputs["embeddings"],
                batch["node_idx"]
            )
            total_loss = ce_loss + self.config.model.lambda_ * cont_loss
        else:
            total_loss = ce_loss
        
        return total_loss
    
    def _contrastive_loss(
        self, 
        embeddings: torch.Tensor, 
        node_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Node embeddings [batch_size, hidden_size]
            node_ids: Node IDs [batch_size]
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.t())
        
        # Create mask for positive pairs (same node_id)
        labels = node_ids.unsqueeze(1)
        mask = (labels == labels.t()).float()
        mask.fill_diagonal_(0)  # Remove self-similarity
        
        # Positive similarity (same node)
        pos_sim = (similarity * mask).sum(1) / mask.sum(1).clamp(min=1)
        
        # Negative similarity (different nodes)
        neg_mask = 1 - mask - torch.eye(len(node_ids), device=embeddings.device)
        neg_sim = similarity.masked_fill(mask.bool() | torch.eye(len(node_ids), device=embeddings.device).bool(), -1e9)
        
        # Get hardest negatives
        hard_neg_sim = neg_sim.topk(min(self.config.model.max_neg, neg_sim.size(1)), dim=1)[0]
        
        # Contrastive loss with margin
        margin = 0.5
        loss = F.relu(hard_neg_sim - pos_sim.unsqueeze(1) + margin).mean()
        
        return loss


def create_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: Config
) -> BaseTrainer:
    """
    Factory function to create appropriate trainer.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: AMP gradient scaler
        config: Training configuration
        
    Returns:
        Trainer instance (PlainTrainer or ContrastiveTrainer)
    """
    if config.model.lambda_ > 0:
        return ContrastiveTrainer(model, optimizer, scheduler, scaler, config)
    else:
        return PlainTrainer(model, optimizer, scheduler, scaler, config)