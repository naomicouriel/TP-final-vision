"""
Training callbacks for checkpointing and early stopping.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Callback:
    """Base callback class."""
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        pass

class ModelCheckpoint(Callback):
    """
    Save model checkpoints based on validation metric.
    """
    def __init__(
        self, 
        dirpath: str, 
        filename: str = "best_model.pt", 
        monitor: str = "val_f1_macro", 
        mode: str = "max"
    ):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.best_score = -np.inf if mode == "max" else np.inf
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            logger.warning(f"Metric {self.monitor} not found in metrics. Available: {metrics.keys()}")
            return
            
        improved = (self.mode == "max" and current_score > self.best_score) or \
                   (self.mode == "min" and current_score < self.best_score)
                   
        if improved:
            self.best_score = current_score
            save_path = self.dirpath / self.filename
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': self.best_score,
                'metrics': metrics
            }, save_path)
            logger.info(f"Model saved to {save_path} (score: {self.best_score:.4f})")

class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.
    """
    def __init__(
        self, 
        monitor: str = "val_loss", 
        min_delta: float = 0.0, 
        patience: int = 5, 
        mode: str = "min"
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        score = metrics.get(self.monitor)
        
        if score is None:
            return
            
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "min" and score > self.best_score - self.min_delta) or \
             (self.mode == "max" and score < self.best_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {epoch} epochs.")
        else:
            self.best_score = score
            self.counter = 0
