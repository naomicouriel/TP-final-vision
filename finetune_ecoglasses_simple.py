"""
Simplified fine-tuning script that focuses on improving ecoglasses detection
by training only on ecoglasses images using a one-vs-all approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from typing import List
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

def prepare_ecoglasses_only_data(ecoglasses_dir: str, train_ratio: float = 0.8):
    """
    Prepare data using only ecoglasses images.
    """
    from sklearn.model_selection import train_test_split
    
    # Get ecoglasses images
    ecoglasses_path = Path(ecoglasses_dir)
    ecoglasses_images = sorted(list(ecoglasses_path.glob('*.jpg')))
    
    if not ecoglasses_images:
        raise ValueError(f"No images found in {ecoglasses_dir}")
    
    # All are ecoglasses (class index 1)
    all_labels = [1] * len(ecoglasses_images)  # 1 = ecoglasses
    
    print(f"Loaded {len(ecoglasses_images)} ecoglasses images")
    
    # Split train/val
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        ecoglasses_images, all_labels, train_size=train_ratio, random_state=42
    )
    
    # Create datasets with strong augmentation for training
    train_dataset = SimpleImageDataset(train_imgs, train_labels, get_train_transforms())
    val_dataset = SimpleImageDataset(val_imgs, val_labels, get_inference_transforms())
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"\nDataset prepared:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def fine_tune_model(
    model_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 15,
    learning_rate: float = 0.00005,
    device: str = 'cpu'
):
    """
    Fine-tune the model focusing on ecoglasses recognition.
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
    
    # Only train the last layers of the classifier
    print("Freezing most layers, training only final classifier layers...")
    for name, param in model.named_parameters():
        # Freeze everything except the final classifier
        if 'classifier.3' in name:  # Only train the final linear layer
            param.requires_grad = True
            print(f"  Training: {name}")
        else:
            param.requires_grad = False
    
    # Use weighted loss to emphasize ecoglasses
    # Create a weight vector that gives more importance to ecoglasses
    class_weights = torch.FloatTensor([1.0, 3.0, 1.0, 1.0]).to(device)  # Higher weight for ecoglasses
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nStarting fine-tuning for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
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

def main():
    print("="*60)
    print("FINE-TUNING FOR ECOGLASSES RECOGNITION")
    print("="*60)
    print("\nThis script will:")
    print("  1. Load your pre-trained model")
    print("  2. Train only the final layer on ecoglasses images")
    print("  3. Use weighted loss to boost ecoglasses recognition")
    print("  4. Take ~5 minutes")
    print()
    
    # Prepare data
    print("Preparing ecoglasses data...")
    train_loader, val_loader = prepare_ecoglasses_only_data(
        ecoglasses_dir='data/custom/ecoglasses_bandeja_jpg',
        train_ratio=0.8
    )
    
    # Fine-tune
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    model, history = fine_tune_model(
        model_path='outputs/checkpoints/best_model.pt',
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=15,
        learning_rate=0.00005,
        device=device
    )
    
    print("\n✅ Fine-tuning complete!")
    print("\nNext steps:")
    print("  1. Test the fine-tuned model:")
    print("     python test_finetuned_model.py")
    print("\n  2. If results are better, use the fine-tuned model:")
    print("     cp outputs/checkpoints/best_model_finetuned.pt outputs/checkpoints/best_model.pt")
    print("\n  3. Then test in your notebook with real-time camera!")

if __name__ == "__main__":
    main()
