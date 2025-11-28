"""
Data preprocessing utilities for recycling classification.
Handles dataset downloading, merging, and splitting.
"""
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import kagglehub
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_kaggle_dataset(dataset_name: str, target_dir: str) -> Path:
    """
    Download dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., "zlatan599/garbage-dataset-classification")
        target_dir: Directory to move the dataset to
        
    Returns:
        Path to the downloaded dataset
    """
    logger.info(f"Downloading {dataset_name}...")
    try:
        # Download latest version
        path = kagglehub.dataset_download(dataset_name)
        logger.info(f"Dataset downloaded to cache: {path}")
        
        # Move to target directory if it doesn't exist
        target_path = Path(target_dir)
        if not target_path.exists():
            target_path.mkdir(parents=True, exist_ok=True)
            # Copy contents
            for item in os.listdir(path):
                s = os.path.join(path, item)
                d = os.path.join(target_path, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            logger.info(f"Dataset moved to {target_path}")
        else:
            logger.info(f"Target directory {target_path} already exists. Skipping move.")
            
        return target_path
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

def get_image_files(directory: Path, extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp')) -> List[Path]:
    """Recursively find all image files in a directory."""
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                images.append(Path(root) / file)
    return images

def create_dataset_csv(
    raw_data_dir: str,
    custom_data_dir: str,
    output_dir: str,
    class_mapping: Dict[str, List[str]],
    split_ratios: Dict[str, float] = {'train': 0.7, 'val': 0.15, 'test': 0.15},
    random_seed: int = 42
):
    """
    Create train/val/test CSVs from raw and custom data.
    
    Args:
        raw_data_dir: Path to Kaggle dataset
        custom_data_dir: Path to custom ecoglasses dataset
        output_dir: Directory to save CSVs
        class_mapping: Dictionary mapping new classes to original classes
        split_ratios: Train/val/test split ratios
        random_seed: Random seed for reproducibility
    """
    raw_path = Path(raw_data_dir)
    custom_path = Path(custom_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    # Invert class mapping for easier lookup: old_class -> new_class
    old_to_new = {}
    for new_class, old_classes in class_mapping.items():
        for old_class in old_classes:
            old_to_new[old_class] = new_class
            
    logger.info(f"Class mapping: {old_to_new}")
    
    # Process Kaggle dataset
    if raw_path.exists():
        logger.info(f"Processing raw data from {raw_path}")
        # Assuming structure: raw_path / class_name / image.jpg
        for class_dir in raw_path.iterdir():
            if class_dir.is_dir():
                original_label = class_dir.name.lower()
                
                # Check if this label should be mapped
                if original_label in old_to_new:
                    new_label = old_to_new[original_label]
                    images = get_image_files(class_dir)
                    
                    for img in images:
                        # Store relative path
                        rel_path = img.relative_to(Path("data")) # Assuming data is root
                        all_data.append({
                            'filename': str(rel_path),
                            'label': new_label,
                            'original_label': original_label,
                            'source': 'kaggle'
                        })
    else:
        logger.warning(f"Raw data directory {raw_path} not found.")

    # Process Custom dataset (Ecoglasses)
    if custom_path.exists():
        logger.info(f"Processing custom data from {custom_path}")
        # Assuming structure: custom_path / image.jpg (all are ecoglasses)
        # OR custom_path / class_name / image.jpg
        
        # Check if 'ecoglasses' is in class_mapping keys
        target_class = 'ecoglasses'
        if target_class in class_mapping:
             # Look for images directly in custom_path or in subdirs
            images = get_image_files(custom_path)
            for img in images:
                 # Store relative path
                try:
                    rel_path = img.relative_to(Path("data"))
                except ValueError:
                    # If custom path is not under data, copy it or use absolute (here we assume under data)
                    rel_path = img
                
                all_data.append({
                    'filename': str(rel_path),
                    'label': target_class,
                    'original_label': 'custom_ecoglasses',
                    'source': 'custom'
                })
    else:
        logger.warning(f"Custom data directory {custom_path} not found.")

    if not all_data:
        raise ValueError("No data found! Check paths and class mappings.")
        
    df = pd.DataFrame(all_data)
    logger.info(f"Total images found: {len(df)}")
    logger.info(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Stratified Split
    # First split: Train vs Temp (Val + Test)
    train_ratio = split_ratios['train']
    val_ratio = split_ratios['val']
    test_ratio = split_ratios['test']
    
    # Normalize ratios if they don't sum to 1
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=(1 - train_ratio), 
        stratify=df['label'], 
        random_state=random_seed
    )
    
    # Second split: Val vs Test
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_relative_ratio),
        stratify=temp_df['label'],
        random_state=random_seed
    )
    
    # Save CSVs
    train_csv = output_path / 'train.csv'
    val_csv = output_path / 'val.csv'
    test_csv = output_path / 'test.csv'
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    logger.info(f"Saved splits to {output_path}")
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_csv, val_csv, test_csv

if __name__ == "__main__":
    # Example usage for testing
    # This block will be executed if script is run directly
    pass
