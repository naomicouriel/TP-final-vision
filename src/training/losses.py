"""
Custom loss functions for recycling classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: 'mean', 'sum' or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Logits (B, C)
            targets: Labels (B)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_function(config: dict, device: str = 'cpu') -> nn.Module:
    """
    Factory function to get loss function based on config.
    """
    loss_name = config.get('criterion', 'CrossEntropyLoss')
    
    if loss_name == 'FocalLoss':
        return FocalLoss().to(device)
    elif loss_name == 'CrossEntropyLoss':
        label_smoothing = config.get('label_smoothing', 0.0)
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
