"""
MobileNet model wrapper for recycling classification.
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Literal


class MobileNetClassifier(nn.Module):
    """
    MobileNet wrapper for recycling classification.
    Supports MobileNetV2 and MobileNetV3 architectures.
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        architecture: Literal['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'] = 'mobilenet_v3_large',
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        """
        Args:
            num_classes: Number of output classes
            architecture: MobileNet variant to use
            pretrained: Whether to load ImageNet pretrained weights
            dropout: Dropout rate for final classifier
        """
        super().__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        
        # Load base model
        if architecture == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.mobilenet_v2(weights=weights)
            in_features = self.model.classifier[1].in_features
            
            # Replace classifier
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
            
        elif architecture == 'mobilenet_v3_small':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.mobilenet_v3_small(weights=weights)
            in_features = self.model.classifier[3].in_features
            
            # Replace classifier
            self.model.classifier = nn.Sequential(
                nn.Linear(self.model.classifier[0].out_features, 1024),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(1024, num_classes)
            )
            
        elif architecture == 'mobilenet_v3_large':
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.mobilenet_v3_large(weights=weights)
            in_features = self.model.classifier[3].in_features
            
            # Replace classifier
            self.model.classifier = nn.Sequential(
                nn.Linear(self.model.classifier[0].out_features, 1280),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(1280, num_classes)
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all layers except classifier (for transfer learning)."""
        for param in self.model.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def load_checkpoint(model: MobileNetClassifier, checkpoint_path: str, device: str = 'cpu'):
    """
    Load model from checkpoint.
    
    Args:
        model: MobileNetClassifier instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model and checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


def export_to_onnx(model: MobileNetClassifier, save_path: str, input_size: tuple = (1, 3, 224, 224)):
    """
    Export model to ONNX format.
    
    Args:
        model: MobileNetClassifier instance
        save_path: Path to save ONNX model
        input_size: Input tensor size (batch, channels, height, width)
    """
    model.eval()
    dummy_input = torch.randn(input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to ONNX: {save_path}")


def export_to_torchscript(model: MobileNetClassifier, save_path: str):
    """
    Export model to TorchScript format.
    
    Args:
        model: MobileNetClassifier instance
        save_path: Path to save TorchScript model
    """
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(save_path)
    print(f"Model exported to TorchScript: {save_path}")
