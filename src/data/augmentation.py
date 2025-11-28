"""
Data augmentation and transformation pipelines using Albumentations.
Adapted from original notebook for MobileNet input requirements.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 224):
    """
    Get training augmentation pipeline.
    
    Args:
        image_size: Target image size (default: 224 for MobileNet)
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),  # ImageNet stats
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 224):
    """
    Get validation/test augmentation pipeline.
    
    Args:
        image_size: Target image size (default: 224 for MobileNet)
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def get_inference_transforms(image_size: int = 224):
    """
    Get inference augmentation pipeline (minimal processing).
    
    Args:
        image_size: Target image size (default: 224 for MobileNet)
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            value=(0, 0, 0)
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])
