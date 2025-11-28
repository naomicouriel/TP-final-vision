"""
PyTorch Dataset class for recycling classification.
"""
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Callable


class RecyclingDataset(Dataset):
    """
    Custom Dataset for recycling classification.
    Supports loading from CSV with image paths and labels.
    """
    
    def __init__(
        self,
        csv_path: str,
        root_dir: str = "data/raw",
        transforms: Optional[Callable] = None,
        class2idx: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            csv_path: Path to CSV file with columns ['filename', 'label']
            root_dir: Root directory containing images
            transforms: Albumentations transform pipeline
            class2idx: Class name to index mapping (if None, created from data)
        """
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Create or use provided class mapping
        if class2idx is None:
            unique_classes = sorted(self.df['label'].unique())
            self.class2idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        else:
            self.class2idx = class2idx
        
        self.idx2class = {v: k for k, v in self.class2idx.items()}
        self.num_classes = len(self.class2idx)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        """
        Get item by index.
        
        Returns:
            tuple: (image_tensor, label_idx)
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.root_dir / row['filename']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        # Get label
        label = row['label']
        label_idx = self.class2idx[label]
        
        return image, label_idx
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in dataset."""
        return self.df['label'].value_counts().to_dict()
    
    def get_sample_by_class(self, class_name: str, n: int = 1):
        """Get n random samples from a specific class."""
        class_df = self.df[self.df['label'] == class_name]
        if len(class_df) == 0:
            raise ValueError(f"No samples found for class: {class_name}")
        return class_df.sample(n=min(n, len(class_df)))
