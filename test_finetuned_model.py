"""
Test the fine-tuned model on ecoglasses_bandeja images.
Compare performance with the original model.
"""
from pathlib import Path
from src.inference.predictor import RecyclingPredictor
import numpy as np
from collections import Counter

def test_model(model_path: str, model_name: str):
    """Test a model and return prediction statistics."""
    
    class_mapping = {
        0: 'cardboard_paper',
        1: 'ecoglasses',
        2: 'metal_plastic',
        3: 'trash'
    }
    
    # Initialize predictor
    predictor = RecyclingPredictor(
        model_path=model_path,
        num_classes=4,
        architecture='mobilenet_v3_large',
        device='cpu',
        class_mapping=class_mapping
    )
    
    # Get test images
    ecoglasses_dir = Path('data/custom/ecoglasses_bandeja_jpg')
    image_files = sorted(list(ecoglasses_dir.glob('*.jpg')))
    
    # Run predictions
    predictions = []
    confidences = []
    
    for img_path in image_files:
        result = predictor.predict(str(img_path))
        predictions.append(result['class_name'])
        confidences.append(result['confidence'])
    
    # Calculate statistics
    prediction_counts = Counter(predictions)
    total = len(predictions)
    ecoglasses_count = prediction_counts['ecoglasses']
    ecoglasses_accuracy = (ecoglasses_count / total) * 100
    
    ecoglasses_confidences = [conf for pred, conf in zip(predictions, confidences) 
                             if pred == 'ecoglasses']
    
    return {
        'total': total,
        'ecoglasses_count': ecoglasses_count,
        'accuracy': ecoglasses_accuracy,
        'predictions': prediction_counts,
        'mean_confidence': np.mean(ecoglasses_confidences) if ecoglasses_confidences else 0,
        'confidences': ecoglasses_confidences
    }

def main():
    print("="*70)
    print("COMPARING ORIGINAL VS FINE-TUNED MODEL")
    print("="*70)
    print()
    
    # Test original model
    print("Testing ORIGINAL model...")
    original_stats = test_model(
        'outputs/checkpoints/best_model.pt',
        'Original'
    )
    
    print(f"  Ecoglasses detected: {original_stats['ecoglasses_count']}/{original_stats['total']} "
          f"({original_stats['accuracy']:.1f}%)")
    print(f"  Mean confidence: {original_stats['mean_confidence']:.3f}")
    print()
    
    # Test fine-tuned model
    print("Testing FINE-TUNED model...")
    try:
        finetuned_stats = test_model(
            'outputs/checkpoints/best_model_finetuned.pt',
            'Fine-tuned'
        )
        
        print(f"  Ecoglasses detected: {finetuned_stats['ecoglasses_count']}/{finetuned_stats['total']} "
              f"({finetuned_stats['accuracy']:.1f}%)")
        print(f"  Mean confidence: {finetuned_stats['mean_confidence']:.3f}")
        print()
        
        # Comparison
        print("="*70)
        print("COMPARISON")
        print("="*70)
        
        accuracy_improvement = finetuned_stats['accuracy'] - original_stats['accuracy']
        confidence_improvement = finetuned_stats['mean_confidence'] - original_stats['mean_confidence']
        
        print(f"\nAccuracy improvement: {accuracy_improvement:+.1f}%")
        print(f"Confidence improvement: {confidence_improvement:+.3f}")
        
        print("\nPrediction distribution:")
        print(f"{'Class':<20} {'Original':>15} {'Fine-tuned':>15} {'Change':>15}")
        print("-"*70)
        
        for class_name in ['ecoglasses', 'cardboard_paper', 'metal_plastic', 'trash']:
            orig_count = original_stats['predictions'][class_name]
            fine_count = finetuned_stats['predictions'][class_name]
            change = fine_count - orig_count
            
            print(f"{class_name:<20} {orig_count:>15d} {fine_count:>15d} {change:+15d}")
        
        print()
        
        if accuracy_improvement > 10:
            print("✅ Significant improvement! The fine-tuned model is much better.")
        elif accuracy_improvement > 5:
            print("✅ Good improvement! The fine-tuned model performs better.")
        elif accuracy_improvement > 0:
            print("✓ Slight improvement. The fine-tuned model is a bit better.")
        else:
            print("⚠️ No improvement. The original model was already well-suited for this data.")
        
        print("\nRecommendation:")
        if finetuned_stats['accuracy'] > original_stats['accuracy']:
            print("  Use the fine-tuned model for better ecoglasses_bandeja predictions:")
            print("  cp outputs/checkpoints/best_model_finetuned.pt outputs/checkpoints/best_model.pt")
        else:
            print("  Keep using the original model.")
        
    except FileNotFoundError:
        print("  ❌ Fine-tuned model not found!")
        print("  Run: python finetune_ecoglasses.py")
        print()
        return
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
