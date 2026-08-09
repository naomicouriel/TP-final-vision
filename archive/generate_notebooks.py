import json
import os

def create_notebook(cells, filename):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created {filename}")

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().split('\n')
    }

def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().split('\n')
    }

# ---------------------------------------------------------
# 01_data_preparation.ipynb
# ---------------------------------------------------------
cells_01 = [
    markdown_cell("# Data Preparation\n\nThis notebook handles downloading the dataset, merging custom data, and creating train/val/test splits."),
    code_cell("""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path('..').resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.preprocessing import download_kaggle_dataset, create_dataset_csv
from src.utils.config import Config
"""),
    markdown_cell("## 1. Load Configuration"),
    code_cell("""
config = Config('../configs/mobilenet_config.yaml')
print("Configuration loaded.")
"""),
    markdown_cell("## 2. Download Kaggle Dataset"),
    code_cell("""
# Download dataset
dataset_name = config.data.get('kaggle_dataset')
raw_dir = Path('../data/raw')

print(f"Downloading {dataset_name}...")
download_kaggle_dataset(dataset_name, str(raw_dir))
"""),
    markdown_cell("## 3. Create Splits (Train/Val/Test)\n\nThis step merges the Kaggle dataset with your custom ecoglasses (if any) and creates the CSV files."),
    code_cell("""
# Define paths
custom_dir = Path('../') / config.data.get('custom_ecoglasses_folder')
output_dir = Path('../') / config.data.get('processed_dir')
class_mapping = config.get('class_mapping')
split_ratios = config.data.get('split_ratios')

# Create CSVs
train_csv, val_csv, test_csv = create_dataset_csv(
    raw_data_dir=raw_dir,
    custom_data_dir=custom_dir,
    output_dir=output_dir,
    class_mapping=class_mapping,
    split_ratios=split_ratios
)

print(f"Train CSV: {train_csv}")
print(f"Val CSV: {val_csv}")
print(f"Test CSV: {test_csv}")
""")
]

create_notebook(cells_01, 'notebooks/01_data_preparation.ipynb')

# ---------------------------------------------------------
# 02_train_mobilenet.ipynb
# ---------------------------------------------------------
cells_02 = [
    markdown_cell("# Train MobileNet\n\nThis notebook trains the MobileNet model for recycling classification."),
    code_cell("""
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path
project_root = Path('..').resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import Config
from src.data.dataset import RecyclingDataset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.models.mobilenet import MobileNetClassifier, export_to_onnx, export_to_torchscript
from src.training.trainer import Trainer
from src.training.losses import get_loss_function
from src.training.callbacks import ModelCheckpoint, EarlyStopping
from src.utils.visualization import plot_training_history, show_batch
"""),
    markdown_cell("## 1. Setup"),
    code_cell("""
# Load config
config = Config('../configs/mobilenet_config.yaml')

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create directories
Path('../outputs/checkpoints').mkdir(parents=True, exist_ok=True)
Path('../outputs/logs').mkdir(parents=True, exist_ok=True)
Path('../outputs/exports').mkdir(parents=True, exist_ok=True)
"""),
    markdown_cell("## 2. Data Loading"),
    code_cell("""
# Transforms
img_size = config.data.get('image_size', 224)
train_transforms = get_train_transforms(img_size)
val_transforms = get_val_transforms(img_size)

# Datasets
processed_dir = Path('../') / config.data.get('processed_dir')
train_dataset = RecyclingDataset(processed_dir / 'train.csv', root_dir='../data', transforms=train_transforms)
val_dataset = RecyclingDataset(processed_dir / 'val.csv', root_dir='../data', transforms=val_transforms)

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Classes: {train_dataset.class2idx}")

# Dataloaders
batch_size = config.training.get('batch_size', 32)
num_workers = config.training.get('num_workers', 2)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
"""),
    markdown_cell("## 3. Visualize Batch"),
    code_cell("""
# Show sample batch
images, labels = next(iter(train_loader))
show_batch(images, labels, train_dataset.idx2class)
"""),
    markdown_cell("## 4. Model Initialization"),
    code_cell("""
model = MobileNetClassifier(
    num_classes=config.model.get('num_classes'),
    architecture=config.model.get('architecture'),
    pretrained=config.model.get('pretrained'),
    dropout=config.model.get('dropout')
)
print(f"Model created: {config.model.get('architecture')}")
"""),
    markdown_cell("## 5. Training Setup"),
    code_cell("""
# Loss and Optimizer
criterion = get_loss_function(config.training, device)
optimizer = optim.Adam(
    model.parameters(), 
    lr=float(config.training.get('lr')), 
    weight_decay=float(config.training.get('weight_decay'))
)

# Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=config.training.get('epochs')
)

# Callbacks
callbacks = [
    ModelCheckpoint(dirpath='../outputs/checkpoints', monitor='val_f1_macro', mode='max'),
    EarlyStopping(monitor='val_loss', patience=config.training.get('early_stopping', {}).get('patience', 5))
]

# Trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    config=config._config,
    scheduler=scheduler,
    callbacks=callbacks
)
"""),
    markdown_cell("## 6. Train"),
    code_cell("""
epochs = config.training.get('epochs', 10)
history = trainer.fit(train_loader, val_loader, epochs=epochs)
"""),
    markdown_cell("## 7. Results"),
    code_cell("""
plot_training_history(history)
"""),
    markdown_cell("## 8. Export Model"),
    code_cell("""
# Load best model
best_model_path = '../outputs/checkpoints/best_model.pt'
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Export
export_to_onnx(model, '../outputs/exports/mobilenet.onnx')
export_to_torchscript(model, '../outputs/exports/mobilenet_ts.pt')
""")
]

create_notebook(cells_02, 'notebooks/02_train_mobilenet.ipynb')

# ---------------------------------------------------------
# 03_inference_demo.ipynb
# ---------------------------------------------------------
cells_03 = [
    markdown_cell("# Inference Demo\n\nTest the trained model on images or camera feed."),
    code_cell("""
import sys
from pathlib import Path

# Add project root to path
project_root = Path('..').resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.inference.predictor import RecyclingPredictor
from src.inference.camera import CameraInference
from src.utils.config import Config
"""),
    markdown_cell("## 1. Load Predictor"),
    code_cell("""
config = Config('../configs/mobilenet_config.yaml')
model_path = '../outputs/checkpoints/best_model.pt'

# We need the class mapping from training. 
# Ideally this is saved with the model or config.
# For now we reconstruct it or hardcode it if known.
# In a real scenario, save class_mapping.json during training.
class_mapping = {0: 'cardboard_paper', 1: 'ecoglasses', 2: 'metal_plastic', 3: 'trash'} # Example, verify with training output

predictor = RecyclingPredictor(
    model_path=model_path,
    num_classes=config.model.get('num_classes'),
    architecture=config.model.get('architecture'),
    class_mapping=class_mapping
)
print("Predictor loaded.")
"""),
    markdown_cell("## 2. Single Image Prediction"),
    code_cell("""
# Test on a sample image from test set
# img_path = '../data/processed/test/some_image.jpg'
# result = predictor.predict(img_path)
# print(result)
"""),
    markdown_cell("## 3. Real-time Camera Demo"),
    code_cell("""
# Start camera inference
# Press 'q' to quit window

camera = CameraInference(predictor, camera_id=0)
# camera.run() 
# Uncomment above line to run. Note: cv2.imshow might not work in all notebook environments (e.g. Colab).
# Works best locally.
""")
]

create_notebook(cells_03, 'notebooks/03_inference_demo.ipynb')
