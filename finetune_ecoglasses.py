"""
Fine-tune the model on ecoglasses_bandeja images to improve prediction accuracy.

This script demonstrates transfer learning / fine-tuning without full retraining:
1. Load the pre-trained model
2. Freeze early layers (feature extraction)
3. Train only the final classification layer on new data
4. Use a small learning rate to preserve learned features

This is MUCH faster than full retraining (minutes vs. hours).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple
import json

from src.models.mobilenet import MobileNetClassifier
from src.data.augmentation import get_train_transforms, get_inference_transforms

class SimpleImageDataset(Dataset):
    """Simple dataset for fine-tuning."""
    
    def __init__(self, image_paths: List[Path], labels: List[int], transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        label = self.labels[idx]
        return image, label

def prepare_finetuning_data(
    ecoglasses_dir: str,
    other_classes_data: dict,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare data for fine-tuning.
    
    Args:
        ecoglasses_dir: Directory with new ecoglasses images
        other_classes_data: Dict with paths for other classes (to maintain balance)
        train_ratio: Ratio for train/val split
    """
    from sklearn.model_selection import train_test_split
    
    # Class mapping
    class_to_idx = {
        'cardboard_paper': 0,
        'ecoglasses': 1,
        'metal_plastic': 2,
        'trash': 3
    }
    
    all_images = []
    all_labels = []
    
    # Add ecoglasses images
    ecoglasses_path = Path(ecoglasses_dir)
    ecoglasses_images = list(ecoglasses_path.glob('*.jpg'))
    
    # Filter out any non-existent files
    ecoglasses_images = [img for img in ecoglasses_images if img.exists()]
    
    all_images.extend(ecoglasses_images)
    all_labels.extend([class_to_idx['ecoglasses']] * len(ecoglasses_images))
    
    print(f"Loaded {len(ecoglasses_images)} ecoglasses images")
    
    # Add some images from other classes for balance
    for class_name, image_paths in other_classes_data.items():
        if class_name in class_to_idx:
            # Filter out non-existent files
            valid_paths = [p for p in image_paths if Path(p).exists()]
            
            if len(valid_paths) == 0:
                print(f"Warning: No valid images found for class {class_name}")
                continue
                
            num_samples = min(len(valid_paths), len(ecoglasses_images) // 2)
            sampled_paths = np.random.choice(valid_paths, num_samples, replace=False)
            all_images.extend(sampled_paths)
            all_labels.extend([class_to_idx[class_name]] * num_samples)
            print(f"Loaded {num_samples} {class_name} images")
    
    # Split train/val
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        all_images, all_labels, train_size=train_ratio, random_state=42, stratify=all_labels
    )
    
    # Create datasets
    train_dataset = SimpleImageDataset(train_imgs, train_labels, get_train_transforms())
    val_dataset = SimpleImageDataset(val_imgs, val_labels, get_inference_transforms())
    
    # Create dataloaders - set num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"\nDataset prepared:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def fine_tune_model(
    model_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.0001,
    device: str = 'cpu'
):
    """
    Fine-tune the model on new data.
    
    Args:
        model_path: Path to pre-trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate (small for fine-tuning)
        device: Device to train on
    """
    # Load pre-trained model
    print("\nLoading pre-trained model...")
    model = MobileNetClassifier(
        num_classes=4,
        architecture='mobilenet_v3_large',
        pretrained=False
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    # Freeze early layers - only train the classifier
    print("Freezing feature extraction layers...")
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'fc' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"  Training: {name}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nStarting fine-tuning for {num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = 'outputs/checkpoints/best_model_finetuned.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
        
        print()
    
    print("="*60)
    print(f"Fine-tuning complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Fine-tuned model saved to: outputs/checkpoints/best_model_finetuned.pt")
    
    # Save training history
    with open('outputs/checkpoints/finetuning_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history

def get_sample_images_from_training_data(csv_path: str, num_per_class: int = 50) -> dict:
    """Get sample images from training data for balancing."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    samples = {}
    for class_name in ['cardboard_paper', 'metal_plastic', 'trash']:
        class_df = df[df['label'] == class_name]
        sample_df = class_df.sample(n=min(num_per_class, len(class_df)), random_state=42)
        # Convert relative paths to absolute paths
        samples[class_name] = [Path(f).resolve() for f in sample_df['filename'].values]
    
    return samples

def main():
    print("="*60)
    print("FINE-TUNING MOBILENET ON ECOGLASSES_BANDEJA")
    print("="*60)
    print("\nThis will:")
    print("  1. Load your pre-trained model")
    print("  2. Freeze feature extraction layers")
    print("  3. Train only the classifier on new data")
    print("  4. Take ~5-10 minutes (much faster than full retraining!)")
    print()
    
    # Prepare data
    print("Preparing data...")
    other_classes_data = get_sample_images_from_training_data(
        'data/processed/train.csv',
        num_per_class=50
    )
    
    train_loader, val_loader = prepare_finetuning_data(
        ecoglasses_dir='data/custom/ecoglasses_bandeja_jpg',
        other_classes_data=other_classes_data,
        train_ratio=0.8
    )
    
    # Fine-tune
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    model, history = fine_tune_model(
        model_path='outputs/checkpoints/best_model.pt',
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=0.0001,
        device=device
    )
    
    print("\n✅ Fine-tuning complete!")
    print("\nNext steps:")
    print("  1. Test the fine-tuned model:")
    print("     python test_finetuned_model.py")
    print("\n  2. If satisfied, replace the original model:")
    print("     cp outputs/checkpoints/best_model_finetuned.pt outputs/checkpoints/best_model.pt")

if __name__ == "__main__":
    main()
