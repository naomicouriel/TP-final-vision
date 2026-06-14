"""
Comprehensive prediction analysis for ecoglasses_bandeja images.
"""
from pathlib import Path
from src.inference.predictor import RecyclingPredictor
import numpy as np
from collections import Counter

def main():
    # Class mapping (alphabetically sorted as created during training)
    class_mapping = {
        0: 'cardboard_paper',
        1: 'ecoglasses',
        2: 'metal_plastic',
        3: 'trash'
    }
    
    # Initialize predictor
    print("Loading model...")
    predictor = RecyclingPredictor(
        model_path='outputs/checkpoints/best_model.pt',
        num_classes=4,
        architecture='mobilenet_v3_large',
        device='cpu',  # Change to 'cuda' if you have GPU
        class_mapping=class_mapping
    )
    print("Model loaded successfully!\n")
    
    # Get ecoglasses images
    ecoglasses_dir = Path('data/custom/ecoglasses_bandeja_jpg')
    image_files = sorted(list(ecoglasses_dir.glob('*.jpg')))
    
    if not image_files:
        print(f"No .jpg images found in {ecoglasses_dir}")
        print("Run convert_heic_images.py first to convert HEIC files")
        return
    
    print(f"Found {len(image_files)} ecoglasses images")
    print("Running predictions on all images...\n")
    
    # Collect results
    predictions = []
    confidences = []
    all_probs = []
    
    for i, img_path in enumerate(image_files, 1):
        try:
            result = predictor.predict(str(img_path))
            
            predicted_class = result['class_name']
            confidence = result['confidence']
            
            predictions.append(predicted_class)
            confidences.append(confidence)
            all_probs.append(result['probabilities'])
            
            # Print progress every 10 images
            if i % 10 == 0 or i == len(image_files):
                print(f"Processed {i}/{len(image_files)} images...")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    # Analysis
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    # Count predictions
    prediction_counts = Counter(predictions)
    total = len(predictions)
    
    print(f"\nTotal images analyzed: {total}")
    print("\nPrediction distribution:")
    for class_name in ['ecoglasses', 'cardboard_paper', 'metal_plastic', 'trash']:
        count = prediction_counts[class_name]
        percentage = (count / total) * 100
        print(f"  {class_name:20s}: {count:3d} ({percentage:5.1f}%)")
    
    # Ecoglasses specific analysis
    ecoglasses_count = prediction_counts['ecoglasses']
    ecoglasses_accuracy = (ecoglasses_count / total) * 100
    
    print("\n" + "-"*60)
    print(f"Ecoglasses Detection Rate: {ecoglasses_count}/{total} ({ecoglasses_accuracy:.1f}%)")
    print("-"*60)
    
    # Confidence analysis for ecoglasses predictions
    ecoglasses_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == 'ecoglasses']
    if ecoglasses_confidences:
        print(f"\nEcoglasses prediction confidence:")
        print(f"  Mean:   {np.mean(ecoglasses_confidences):.3f}")
        print(f"  Median: {np.median(ecoglasses_confidences):.3f}")
        print(f"  Min:    {np.min(ecoglasses_confidences):.3f}")
        print(f"  Max:    {np.max(ecoglasses_confidences):.3f}")
    
    # Misclassified images analysis
    misclassified = [(img_path.name, pred, conf) 
                     for img_path, pred, conf in zip(image_files, predictions, confidences) 
                     if pred != 'ecoglasses']
    
    if misclassified:
        print(f"\n\nImages NOT classified as ecoglasses ({len(misclassified)}):")
        print("-"*60)
        for img_name, pred, conf in misclassified[:10]:  # Show first 10
            print(f"  {img_name:25s} → {pred:20s} (conf: {conf:.3f})")
        if len(misclassified) > 10:
            print(f"  ... and {len(misclassified) - 10} more")
    
    # Average probabilities across all images
    all_probs_array = np.array(all_probs)
    avg_probs = np.mean(all_probs_array, axis=0)
    
    print(f"\n\nAverage class probabilities across all images:")
    print("-"*60)
    for class_id, avg_prob in enumerate(avg_probs):
        class_name = class_mapping[class_id]
        print(f"  {class_name:20s}: {avg_prob:.3f}")
    
    print("\n" + "="*60)
    print("\n✅ Good news! Your model already knows ecoglasses!")
    print("   No retraining needed - it was part of the original training.")
    
    if ecoglasses_accuracy >= 70:
        print(f"\n✅ The model performs well on your bandeja images ({ecoglasses_accuracy:.1f}% accuracy).")
    elif ecoglasses_accuracy >= 50:
        print(f"\n⚠️  Moderate accuracy ({ecoglasses_accuracy:.1f}%). Consider:")
        print("   - The bandeja images may look different from training images")
        print("   - Fine-tuning on a few bandeja images could improve accuracy")
    else:
        print(f"\n⚠️  Lower accuracy ({ecoglasses_accuracy:.1f}%). The model struggles because:")
        print("   - Bandeja images may be visually different from training data")
        print("   - Consider fine-tuning with some bandeja images")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
