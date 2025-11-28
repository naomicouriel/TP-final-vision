"""
Evaluation metrics for recycling classification.
"""
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import io

class Metrics:
    """
    Class to compute and store metrics.
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.reset()

    def reset(self):
        """Reset internal state."""
        self.y_true = []
        self.y_pred = []
        self.losses = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float = None):
        """
        Update metrics with batch results.
        
        Args:
            logits: Model output logits (Batch, Num_Classes)
            targets: Ground truth labels (Batch)
            loss: Batch loss value
        """
        preds = torch.argmax(logits, dim=1)
        
        self.y_true.extend(targets.cpu().numpy())
        self.y_pred.extend(preds.cpu().numpy())
        
        if loss is not None:
            self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """
        Compute aggregate metrics.
        
        Returns:
            Dictionary of metrics (accuracy, f1_macro, loss)
        """
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        
        accuracy = np.mean(y_true == y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1,
            'loss': avg_loss
        }

    def get_confusion_matrix(self, class_names: List[str] = None) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(self.y_true, self.y_pred)

    def get_classification_report(self, class_names: List[str] = None) -> str:
        """Get text classification report."""
        return classification_report(self.y_true, self.y_pred, target_names=class_names)

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return fig
