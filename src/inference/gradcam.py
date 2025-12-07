"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for MobileNetV3.
Visualizes which regions of the image the model focuses on for predictions.
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, Optional


class GradCAM:
    """
    Grad-CAM implementation for visualization.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The trained model
            target_layer: The layer to compute gradients on (e.g., last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            Heatmap as numpy array (H, W) in range [0, 1]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # (H, W)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def overlay_heatmap(
        self, 
        image: np.ndarray, 
        heatmap: np.ndarray, 
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original RGB image (H, W, 3)
            heatmap: Grad-CAM heatmap (h, w)
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap
            
        Returns:
            Overlayed image (H, W, 3)
        """
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to uint8 and apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            colormap
        )
        
        # Convert to RGB (OpenCV uses BGR)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed


def get_target_layer_mobilenet(model):
    """
    Get the appropriate target layer for MobileNetV3.
    
    Args:
        model: MobileNetClassifier instance
        
    Returns:
        Target layer for Grad-CAM (last convolutional layer)
    """
    # For MobileNetV3, use the last layer of features
    # model.model.features[-1] is typically the last conv block
    return model.model.features[-1]


class GradCAMPredictor:
    """
    Predictor wrapper with Grad-CAM visualization.
    """
    
    def __init__(self, predictor):
        """
        Args:
            predictor: RecyclingPredictor instance
        """
        self.predictor = predictor
        
        # Setup Grad-CAM
        target_layer = get_target_layer_mobilenet(self.predictor.model)
        self.gradcam = GradCAM(self.predictor.model, target_layer)
    
    def predict_with_cam(self, image: np.ndarray) -> Tuple[dict, np.ndarray]:
        """
        Run prediction with Grad-CAM visualization.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            (prediction_dict, overlayed_image)
        """
        # Get prediction
        result = self.predictor.predict(image)
        
        # Preprocess for Grad-CAM
        tensor = self.predictor.preprocess(image)
        
        # Generate heatmap
        heatmap = self.gradcam.generate_cam(tensor, target_class=result['class_id'])
        
        # Overlay heatmap on original image
        overlayed = self.gradcam.overlay_heatmap(image, heatmap, alpha=0.4)
        
        return result, overlayed
