"""
Advanced Logging System with Rich Terminal Output.

This module provides a comprehensive logging solution with:
- Beautiful rich-formatted console output
- File logging with rotation
- Progress tracking and live displays
- System information logging
- Training metrics visualization
"""

import os
import sys
import logging
import platform
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
from contextlib import contextmanager

import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TimeElapsedColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich.align import Align
from rich import box
from rich.tree import Tree
from rich.status import Status


class MLLogger:
    """
    Advanced logger for machine learning projects with rich formatting.
    
    Features:
    - Rich console output with colors and formatting
    - File logging with automatic rotation
    - Progress bars and live status updates
    - System information display
    - Training metrics visualization
    - Hierarchical logging structure
    """
    
    def __init__(
        self, 
        name: str = __name__,
        log_level: int = logging.INFO,
        console_width: Optional[int] = None
    ):
        """
        Initialize the ML logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            console_width: Console width (auto-detected if None)
        """
        self.name = name
        self.console = Console(width=console_width)
        self.logger = self._setup_logger(name, log_level)
        self.log_level = log_level
        self._log_file_path = None
        
        # Progress tracking
        self._progress_tasks = {}
        self._current_progress = None
        
    def _setup_logger(self, name: str, log_level: int) -> logging.Logger:
        """Setup the base logger with rich handler."""
        logger = logging.getLogger(name)
        
        # Prevent duplicate handlers
        if logger.handlers:
            logger.handlers.clear()
            
        logger.setLevel(log_level)
        
        # Rich console handler
        rich_handler = RichHandler(
            console=self.console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
            tracebacks_show_locals=True,
            omit_repeated_times=False
        )
        
        # Custom format for rich handler
        rich_format = "%(message)s"
        rich_handler.setFormatter(logging.Formatter(rich_format))
        rich_handler.setLevel(log_level)
        
        logger.addHandler(rich_handler)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        return logger
    
    def setup_file_logging(
        self, 
        log_dir: Union[str, Path], 
        prefix: str = "training",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> Path:
        """
        Setup file logging with rotation.
        
        Args:
            log_dir: Directory to save log files
            prefix: Prefix for log file names
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            
        Returns:
            Path to the log file
        """
        from logging.handlers import RotatingFileHandler
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{prefix}_{timestamp}.log"
        
        # Setup rotating file handler
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File gets everything
        
        # Detailed format for file logging
        file_format = (
            "[%(asctime)s] %(levelname)-8s "
            "[%(name)s:%(filename)s:%(lineno)d] %(message)s"
        )
        file_handler.setFormatter(logging.Formatter(file_format))
        
        self.logger.addHandler(file_handler)
        self._log_file_path = log_file
        
        self.info(f"📁 File logging enabled: {log_file}")
        return log_file
    
    # ─────────────────── Basic Logging Methods ───────────────────
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(f"[dim]🐛 {message}[/dim]", **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(f"ℹ️  {message}", **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(f"⚠️  [yellow]{message}[/yellow]", **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(f"❌ [red]{message}[/red]", **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message."""
        self.logger.info(f"✅ [green]{message}[/green]", **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(f"🚨 [bold red]{message}[/bold red]", **kwargs)
    
    # ─────────────────── Display Methods ───────────────────
    
    def print_banner(
        self, 
        title: str, 
        subtitle: str = "", 
        style: str = "cyan",
        box_style = box.DOUBLE
    ):
        """Print a beautiful banner."""
        if subtitle:
            content = f"[bold {style}]{title}[/bold {style}]\n[dim]{subtitle}[/dim]"
        else:
            content = f"[bold {style}]{title}[/bold {style}]"
            
        banner = Panel(
            Align.center(content),
            box=box_style,
            border_style=style,
            padding=(1, 2)
        )
        self.console.print(banner)
    
    def print_section(self, title: str, style: str = "blue"):
        """Print a section header."""
        self.console.print(f"\n[bold {style}]{'─' * 20} {title} {'─' * 20}[/bold {style}]")
    
    def print_config_table(
        self, 
        config: Dict[str, Any], 
        title: str = "Configuration",
        columns: int = 2
    ):
        """Print configuration as a beautiful table."""
        if not config:
            self.warning("No configuration to display")
            return
            
        # Split config into columns if it's large
        items = list(config.items())
        if len(items) > 20 and columns > 1:
            # Create multiple tables side by side
            chunk_size = (len(items) + columns - 1) // columns
            tables = []
            
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                table = Table(box=box.SIMPLE, show_header=False)
                table.add_column("Parameter", style="cyan", no_wrap=True)
                table.add_column("Value", style="magenta")
                
                for key, value in chunk:
                    value_str = self._format_value(value)
                    table.add_row(key, value_str)
                tables.append(table)
            
            # Display tables side by side
            self.console.print(Panel(
                Columns(tables, equal=True, expand=True),
                title=f"[bold blue]{title}[/bold blue]",
                border_style="blue"
            ))
        else:
            # Single table
            table = Table(title=title, box=box.ROUNDED, title_style="bold blue")
            table.add_column("Parameter", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")
            
            for key, value in items:
                value_str = self._format_value(value)
                table.add_row(key, value_str)
            
            self.console.print(table)
    
    def print_metrics_table(
        self, 
        metrics: Dict[str, Union[float, int, str]], 
        title: str = "Metrics"
    ):
        """Print metrics as a beautiful table."""
        if not metrics:
            self.warning("No metrics to display")
            return
            
        table = Table(title=title, box=box.SIMPLE, title_style="bold green")
        table.add_column("Metric", style="green")
        table.add_column("Value", style="bold yellow", justify="right")
        
        for metric, value in metrics.items():
            value_str = self._format_metric_value(value)
            table.add_row(metric, value_str)
        
        self.console.print(table)
    
    def print_system_info(self):
        """Print comprehensive system information."""
        system_info = self._get_system_info()
        self.print_config_table(system_info, "🖥️  System Information")
    
    def print_model_summary(self, model, input_shape: Optional[tuple] = None):
        """Print model architecture summary."""
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                "Model Class": model.__class__.__name__,
                "Total Parameters": f"{total_params:,}",
                "Trainable Parameters": f"{trainable_params:,}",
                "Non-trainable Parameters": f"{total_params - trainable_params:,}",
                "Memory Usage (MB)": f"{total_params * 4 / 1024 / 1024:.2f}"  # Assuming float32
            }
            
            if input_shape:
                model_info["Input Shape"] = str(input_shape)
            
            self.print_config_table(model_info, "🧠 Model Summary")
    
    def print_tree(self, data: Dict[str, Any], title: str = "Data Structure"):
        """Print hierarchical data as a tree."""
        tree = Tree(f"[bold blue]{title}[/bold blue]")
        self._build_tree(tree, data)
        self.console.print(tree)
    
    def _build_tree(self, tree, data, max_depth: int = 3, current_depth: int = 0):
        """Recursively build tree structure."""
        if current_depth >= max_depth:
            return
            
        for key, value in data.items():
            if isinstance(value, dict):
                branch = tree.add(f"[cyan]{key}[/cyan]")
                self._build_tree(branch, value, max_depth, current_depth + 1)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                branch = tree.add(f"[cyan]{key}[/cyan] ({len(value)} items)")
                for i, item in enumerate(value[:3]):  # Show first 3 items
                    branch.add(f"[dim]{i}: {str(item)[:50]}...[/dim]")
                if len(value) > 3:
                    branch.add("[dim]...[/dim]")
            else:
                value_str = self._format_value(value)
                tree.add(f"[cyan]{key}[/cyan]: {value_str}")
    
    # ─────────────────── Progress Tracking ───────────────────
    
    def create_progress(
        self, 
        description: str = "Processing...",
        total: Optional[int] = None,
        show_speed: bool = True
    ) -> Progress:
        """Create a rich progress bar."""
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ]
        
        if total:
            columns.append(MofNCompleteColumn())
        
        columns.extend([
            TimeElapsedColumn(),
            TimeRemainingColumn() if total else TimeElapsedColumn(),
        ])
        
        progress = Progress(*columns, console=self.console)
        self._current_progress = progress
        return progress
    
    @contextmanager
    def progress_context(self, description: str, total: Optional[int] = None):
        """Context manager for progress tracking."""
        progress = self.create_progress(description, total)
        task = progress.add_task(description, total=total)
        
        try:
            with progress:
                yield progress, task
        finally:
            self._current_progress = None
    
    @contextmanager
    def status(self, message: str, spinner: str = "dots"):
        """Context manager for status display."""
        with Status(message, spinner=spinner, console=self.console) as status:
            yield status
    
    # ─────────────────── Utility Methods ───────────────────
    
    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, bool):
            return "✓" if value else "✗"
        elif isinstance(value, int):
            return f"{value:,}"
        elif isinstance(value, float):
            if abs(value) >= 1000:
                return f"{value:,.2f}"
            elif abs(value) >= 1:
                return f"{value:.4f}"
            else:
                return f"{value:.6f}"
        elif isinstance(value, (list, tuple)):
            if len(value) <= 3:
                return str(value)
            else:
                return f"[{len(value)} items]"
        elif isinstance(value, dict):
            return f"{{...}} ({len(value)} keys)"
        elif value is None:
            return "[dim]None[/dim]"
        else:
            str_val = str(value)
            return str_val if len(str_val) <= 50 else str_val[:47] + "..."
    
    def _format_metric_value(self, value: Union[float, int, str]) -> str:
        """Format a metric value for display."""
        if isinstance(value, float):
            if 0 <= value <= 1:
                return f"{value:.4f}"
            else:
                return f"{value:.6f}"
        elif isinstance(value, int):
            return f"{value:,}"
        else:
            return str(value)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information."""
        system_info = {
            "Python Version": platform.python_version(),
            "Platform": platform.platform(),
            "Architecture": platform.architecture()[0],
            "Processor": platform.processor() or "Unknown",
            "CPU Count": os.cpu_count(),
        }
        
        # PyTorch information
        if 'torch' in sys.modules:
            system_info.update({
                "PyTorch Version": torch.__version__,
                "CUDA Available": torch.cuda.is_available(),
            })
            
            if torch.cuda.is_available():
                system_info.update({
                    "CUDA Version": torch.version.cuda,
                    "cuDNN Version": torch.backends.cudnn.version(),
                    "GPU Count": torch.cuda.device_count(),
                })
                
                # GPU details
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    system_info[f"GPU {i}"] = f"{gpu_name} ({gpu_memory:.1f}GB)"
        
        return system_info
    
    def log_exception(self, exc_info=None):
        """Log exception with rich traceback."""
        self.logger.exception("Exception occurred:", exc_info=exc_info)
    
    def get_log_file_path(self) -> Optional[Path]:
        """Get the current log file path."""
        return self._log_file_path


# ─────────────────── Global Logger Management ───────────────────

_global_logger: Optional[MLLogger] = None

def get_logger(
    name: str = __name__, 
    log_level: int = logging.INFO,
    force_new: bool = False
) -> MLLogger:
    """
    Get or create a global logger instance.
    
    Args:
        name: Logger name
        log_level: Logging level
        force_new: Force creation of new logger
        
    Returns:
        MLLogger instance
    """
    global _global_logger
    
    if _global_logger is None or force_new:
        _global_logger = MLLogger(name, log_level)
    
    return _global_logger


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO,
    prefix: str = "training",
    console_width: Optional[int] = None,
    enable_file_logging: bool = True
) -> MLLogger:
    """
    Setup comprehensive logging system.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        prefix: Prefix for log files
        console_width: Console width
        enable_file_logging: Whether to enable file logging
        
    Returns:
        Configured MLLogger instance
    """
    logger = MLLogger(__name__, log_level, console_width)
    
    if enable_file_logging and log_dir:
        logger.setup_file_logging(log_dir, prefix)
    
    # Set as global logger
    global _global_logger
    _global_logger = logger
    
    return logger


# ─────────────────── Convenience Functions ───────────────────

def log_training_start(config: Dict[str, Any], model=None):
    """Log training initialization."""
    logger = get_logger()
    logger.print_banner("🚀 Training Started", "Initializing machine learning pipeline")
    logger.print_system_info()
    logger.print_config_table(config, "🔧 Training Configuration")
    
    if model is not None:
        logger.print_model_summary(model)


def log_training_complete(final_metrics: Dict[str, Any]):
    """Log training completion."""
    logger = get_logger()
    logger.print_banner("🎉 Training Complete", "Final results summary")
    logger.print_metrics_table(final_metrics, "📊 Final Results")


def log_epoch_summary(epoch: int, train_metrics: Dict, val_metrics: Dict):
    """Log epoch summary."""
    logger = get_logger()
    
    all_metrics = {}
    for key, value in train_metrics.items():
        all_metrics[f"Train {key}"] = value
    for key, value in val_metrics.items():
        all_metrics[f"Val {key}"] = value
    
    logger.print_metrics_table(all_metrics, f"📈 Epoch {epoch} Summary")