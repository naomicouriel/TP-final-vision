# Ecoglasses Prediction - Quick Start

## ✅ Your Model Already Works!

Your model was trained with 4 classes including `ecoglasses`, so it can predict your new images **without any retraining**.

## Quick Test (30 seconds)

```bash
# 1. Convert HEIC to JPG
python convert_heic_images.py

# 2. Run predictions with analysis
python analyze_ecoglasses_predictions.py
```

**Current Results**: 58.9% accuracy (56/95 images correctly classified as ecoglasses)

## Batch Prediction with CSV Output

```bash
python batch_predict.py data/custom/ecoglasses_bandeja_jpg --output predictions.csv
```

This creates a CSV file with:
- Filename and path
- Predicted class
- Confidence score
- Probability for each class

**Example output**: `ecoglasses_bandeja_predictions.csv` contains all predictions for easy review in Excel/Google Sheets.

## Use in Your Code

```python
from src.inference.predictor import RecyclingPredictor

class_mapping = {0: 'cardboard_paper', 1: 'ecoglasses', 2: 'metal_plastic', 3: 'trash'}

predictor = RecyclingPredictor(
    model_path='outputs/checkpoints/best_model.pt',
    num_classes=4,
    class_mapping=class_mapping
)

# Single image
result = predictor.predict('path/to/image.jpg')
print(f"{result['class_name']}: {result['confidence']:.2f}")

# Batch processing
from pathlib import Path
for img in Path('images/').glob('*.jpg'):
    result = predictor.predict(str(img))
    print(f"{img.name}: {result['class_name']}")
```

## Want Better Accuracy? (Optional)

Fine-tune the model on your specific images (5-10 minutes):

```bash
# 1. Fine-tune the model
python finetune_ecoglasses.py

# 2. Compare performance
python test_finetuned_model.py

# 3. If better, use it
cp outputs/checkpoints/best_model_finetuned.pt outputs/checkpoints/best_model.pt
```

Fine-tuning adapts the model to your specific ecoglasses_bandeja images and can improve accuracy to 80-90%+.

## All Available Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| Convert images | HEIC → JPG | `python convert_heic_images.py` |
| Quick test | Test 5 images | `python test_ecoglasses_prediction.py` |
| Full analysis | Analyze all 95 images | `python analyze_ecoglasses_predictions.py` |
| Batch predict | Save predictions to CSV | `python batch_predict.py <input_dir>` |
| Fine-tune | Improve accuracy (optional) | `python finetune_ecoglasses.py` |
| Compare models | Test fine-tuned vs original | `python test_finetuned_model.py` |

## Model Classes

- **cardboard_paper** (0): Cardboard and paper
- **ecoglasses** (1): Your ecoglasses ← **Already trained!**
- **metal_plastic** (2): Metal and plastic items
- **trash** (3): Non-recyclable items

## Why No Retraining Needed?

Your model was trained with ecoglasses from the start. Looking at your training data:

```bash
$ cut -d',' -f2 data/processed/train.csv | sort | uniq
cardboard_paper
ecoglasses      ← Already in training data!
metal_plastic
trash
```

The model already learned ecoglasses features during the original training. You just need to use it!

## Need Help?

See the full guide: [`ECOGLASSES_PREDICTION_GUIDE.md`](./ECOGLASSES_PREDICTION_GUIDE.md)
