"""
Main Trainer class for MobileNet recycling classification.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Any
import time

from src.training.metrics import Metrics
from src.training.callbacks import Callback, ModelCheckpoint, EarlyStopping
from src.utils.logger import TrainingLogger

class Trainer:
    """
    Trainer class for managing the training loop.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str,
        config: dict,
        scheduler: Optional[Any] = None,
        callbacks: List[Callback] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        
        self.metrics = Metrics(device)
        self.logger = TrainingLogger(config['paths']['log_dir'])
        
        # Mixed precision
        self.use_amp = config['training'].get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and 'cuda' in device else None
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1_macro': [], 'val_f1_macro': []
        }

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics.reset()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            self.metrics.update(outputs, labels, loss.item())
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        epoch_metrics = self.metrics.compute()
        return {f"train_{k}": v for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        self.metrics.reset()
        
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.metrics.update(outputs, labels, loss.item())
            
        epoch_metrics = self.metrics.compute()
        return {f"val_{k}": v for k, v in epoch_metrics.items()}

    def fit(self, train_dl: DataLoader, val_dl: DataLoader, epochs: int):
        """
        Main training loop.
        """
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_one_epoch(train_dl, epoch)
            
            # Validate
            val_metrics = self.validate(val_dl, epoch)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Update history
            for k, v in all_metrics.items():
                if k in self.history:
                    self.history[k].append(v)
                elif k.replace('train_', '') in ['loss', 'acc', 'f1_macro']:
                     # Initialize if not present (e.g. for custom metrics)
                     if k not in self.history: self.history[k] = []
                     self.history[k].append(v)

            # Log
            duration = time.time() - start_time
            self.logger.log_metrics(epoch, all_metrics)
            
            # Callbacks
            stop_training = False
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, all_metrics, self.model, self.optimizer)
                if isinstance(callback, EarlyStopping) and callback.early_stop:
                    stop_training = True
            
            if stop_training:
                self.logger.info("Early stopping triggered.")
                break
                
        self.logger.info("Training completed.")
        return self.history
