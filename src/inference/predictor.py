"""
Inference predictor wrapper for MobileNet recycling classification.
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Union, Optional
from PIL import Image

from src.models.mobilenet import MobileNetClassifier
from src.data.augmentation import get_inference_transforms

class RecyclingPredictor:
    """
    Wrapper for running inference with trained MobileNet model.
    """
    def __init__(
        self,
        model_path: str,
        num_classes: int = 4,
        architecture: str = 'mobilenet_v3_large',
        device: str = 'cpu',
        class_mapping: Optional[Dict[int, str]] = None
    ):
        """
        Args:
            model_path: Path to trained model checkpoint (.pt)
            num_classes: Number of classes
            architecture: Model architecture name
            device: Device to run inference on ('cpu' or 'cuda')
            class_mapping: Dictionary mapping class indices to names
        """
        self.device = device
        self.class_mapping = class_mapping
        
        # Load model
        self.model = MobileNetClassifier(
            num_classes=num_classes,
            architecture=architecture,
            pretrained=False  # No need to download weights, we load checkpoint
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(device)
        self.model.eval()
        
        # Setup transforms
        self.transforms = get_inference_transforms()

    def preprocess(self, image: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for model input.
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR if coming from OpenCV, convert to RGB
                # But if user passes RGB numpy array, this might be wrong.
                # Standardizing on expecting RGB numpy array or path.
                # For safety with OpenCV inputs (which are BGR), we might want to ensure RGB.
                # However, Albumentations expects RGB.
                pass
                
        # Apply transforms
        transformed = self.transforms(image=image)
        tensor = transformed['image'].unsqueeze(0) # Add batch dim
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Run prediction on a single image.
        
        Returns:
            Dictionary with 'class_id', 'class_name', 'confidence', 'probabilities'
        """
        tensor = self.preprocess(image)
        
        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1)
        
        conf, pred_idx = torch.max(probs, dim=1)
        pred_idx = pred_idx.item()
        confidence = conf.item()
        
        result = {
            'class_id': pred_idx,
            'confidence': confidence,
            'probabilities': probs.cpu().numpy()[0]
        }
        
        if self.class_mapping:
            result['class_name'] = self.class_mapping.get(pred_idx, str(pred_idx))
            
        return result

    @torch.no_grad()
    def predict_batch(self, images: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """Run prediction on a batch of images."""
        results = []
        for img in images:
            results.append(self.predict(img))
        return results
