"""
Test script to predict ecoglasses from the new bandeja images.
"""
from pathlib import Path
from src.inference.predictor import RecyclingPredictor
import sys

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
    
    # Get ecoglasses images (use JPG converted versions)
    ecoglasses_dir = Path('data/custom/ecoglasses_bandeja_jpg')
    image_files = list(ecoglasses_dir.glob('*.jpg'))
    
    if not image_files:
        print(f"No .jpg images found in {ecoglasses_dir}")
        print("Run convert_heic_images.py first to convert HEIC files")
        return
    
    print(f"Found {len(image_files)} ecoglasses images")
    print("Running predictions on first 5 images...\n")
    
    # Test on first few images
    correct_predictions = 0
    total_tested = 0
    
    for img_path in image_files[:5]:
        try:
            result = predictor.predict(str(img_path))
            
            predicted_class = result['class_name']
            confidence = result['confidence']
            
            print(f"Image: {img_path.name}")
            print(f"  Predicted: {predicted_class} (confidence: {confidence:.3f})")
            print(f"  All probabilities:")
            for class_id, prob in enumerate(result['probabilities']):
                print(f"    {class_mapping[class_id]}: {prob:.3f}")
            print()
            
            if predicted_class == 'ecoglasses':
                correct_predictions += 1
            total_tested += 1
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            print("  (HEIC files may need conversion to JPG/PNG)\n")
            continue
    
    if total_tested > 0:
        accuracy = correct_predictions / total_tested * 100
        print(f"\nResults: {correct_predictions}/{total_tested} correctly predicted as ecoglasses ({accuracy:.1f}%)")
    else:
        print("\nNote: HEIC images need to be converted to JPG/PNG format.")
        print("You can convert them using: sips -s format jpeg <input.heic> --out <output.jpg>")

if __name__ == "__main__":
    main()
