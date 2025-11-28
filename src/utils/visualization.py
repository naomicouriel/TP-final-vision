"""
Visualization utilities for model predictions and data analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple, Optional

def show_batch(
    images: torch.Tensor, 
    labels: torch.Tensor, 
    idx2class: dict, 
    preds: Optional[torch.Tensor] = None,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    n_max: int = 8
):
    """
    Display a batch of images with labels.
    
    Args:
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (B)
        idx2class: Mapping from index to class name
        preds: Optional batch of predictions (B)
        mean: Normalization mean to denormalize
        std: Normalization std to denormalize
        n_max: Maximum number of images to show
    """
    batch_size = len(images)
    n = min(batch_size, n_max)
    
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]
    
    # Denormalize
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    for i in range(n):
        ax = axes[i]
        
        # Un-normalize and convert to numpy
        img = images[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        true_label = idx2class[labels[i].item()]
        title = f"True: {true_label}"
        
        if preds is not None:
            pred_label = idx2class[preds[i].item()]
            color = 'green' if labels[i] == preds[i] else 'red'
            title += f"\nPred: {pred_label}"
            ax.set_title(title, color=color, fontsize=10)
        else:
            ax.set_title(title, fontsize=10)
            
        ax.axis('off')
        
    # Hide empty subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    return fig

def plot_training_history(history: dict):
    """
    Plot training and validation metrics history.
    
    Args:
        history: Dictionary containing lists of metrics
                 e.g., {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Loss vs Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy/F1 plot
    metric_key = 'train_f1_macro' if 'train_f1_macro' in history else 'train_acc'
    val_metric_key = 'val_f1_macro' if 'val_f1_macro' in history else 'val_acc'
    label = 'F1 Macro' if 'f1' in metric_key else 'Accuracy'
    
    if metric_key in history:
        ax2.plot(epochs, history[metric_key], 'b-', label=f'Training {label}')
    if val_metric_key in history:
        ax2.plot(epochs, history[val_metric_key], 'r-', label=f'Validation {label}')
    ax2.set_title(f'{label} vs Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(label)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig
