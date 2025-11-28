"""
Logging utilities for training monitoring.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str, 
    log_dir: str, 
    filename: str = "train.log", 
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a logger that writes to console and file.
    
    Args:
        name: Logger name
        log_dir: Directory to save log file
        filename: Log filename
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

class TrainingLogger:
    """
    Wrapper for logging training progress.
    """
    def __init__(self, log_dir: str):
        self.logger = setup_logger("training", log_dir)
        
    def log_metrics(self, epoch: int, metrics: dict, stage: str = "Train"):
        """
        Log metrics for an epoch.
        """
        msg = f"[{stage}] Epoch {epoch} | "
        msg += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(msg)
        
    def info(self, msg: str):
        self.logger.info(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)
