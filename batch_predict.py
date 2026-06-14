"""
Batch prediction script for ecoglasses images.
Saves predictions to a CSV file for easy review.
"""
from pathlib import Path
from src.inference.predictor import RecyclingPredictor
import pandas as pd
from datetime import datetime
import argparse

def predict_and_save(
    input_dir: str,
    output_csv: str = None,
    model_path: str = 'outputs/checkpoints/best_model.pt',
    device: str = 'cpu'
):
    """
    Run predictions on all images in a directory and save to CSV.
    
    Args:
        input_dir: Directory containing images to predict
        output_csv: Output CSV path (default: auto-generated)
        model_path: Path to model checkpoint
        device: Device to use ('cpu' or 'cuda')
    """
    # Auto-generate output filename if not provided
    if output_csv is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv = f'predictions_{timestamp}.csv'
    
    # Class mapping
    class_mapping = {
        0: 'cardboard_paper',
        1: 'ecoglasses',
        2: 'metal_plastic',
        3: 'trash'
    }
    
    # Initialize predictor
    print(f"Loading model from {model_path}...")
    predictor = RecyclingPredictor(
        model_path=model_path,
        num_classes=4,
        architecture='mobilenet_v3_large',
        device=device,
        class_mapping=class_mapping
    )
    print("Model loaded!\n")
    
    # Get images
    input_path = Path(input_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(ext))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No images found in {input_dir}")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Running predictions...\n")
    
    # Run predictions
    results = []
    
    for i, img_path in enumerate(image_files, 1):
        try:
            result = predictor.predict(str(img_path))
            
            # Create result row
            row = {
                'filename': img_path.name,
                'filepath': str(img_path),
                'predicted_class': result['class_name'],
                'confidence': result['confidence'],
                'prob_cardboard_paper': result['probabilities'][0],
                'prob_ecoglasses': result['probabilities'][1],
                'prob_metal_plastic': result['probabilities'][2],
                'prob_trash': result['probabilities'][3],
            }
            results.append(row)
            
            # Progress indicator
            if i % 10 == 0 or i == len(image_files):
                print(f"  Processed {i}/{len(image_files)} images...")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Predictions saved to: {output_csv}")
    
    # Print summary
    print("\nPrediction Summary:")
    print("-" * 50)
    summary = df['predicted_class'].value_counts()
    for class_name, count in summary.items():
        percentage = (count / len(df)) * 100
        avg_conf = df[df['predicted_class'] == class_name]['confidence'].mean()
        print(f"  {class_name:20s}: {count:3d} ({percentage:5.1f}%) - avg conf: {avg_conf:.3f}")
    
    print("\nTop 5 most confident predictions:")
    print("-" * 50)
    top5 = df.nlargest(5, 'confidence')
    for _, row in top5.iterrows():
        print(f"  {row['filename']:30s} → {row['predicted_class']:15s} ({row['confidence']:.3f})")
    
    print("\nTop 5 least confident predictions:")
    print("-" * 50)
    bottom5 = df.nsmallest(5, 'confidence')
    for _, row in bottom5.iterrows():
        print(f"  {row['filename']:30s} → {row['predicted_class']:15s} ({row['confidence']:.3f})")
    
    print(f"\n📊 Full results available in: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Batch prediction for recycling classification')
    parser.add_argument('input_dir', type=str, help='Directory containing images to predict')
    parser.add_argument('--output', '-o', type=str, default=None, 
                       help='Output CSV path (default: auto-generated)')
    parser.add_argument('--model', '-m', type=str, 
                       default='outputs/checkpoints/best_model.pt',
                       help='Model checkpoint path')
    parser.add_argument('--device', '-d', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    predict_and_save(
        input_dir=args.input_dir,
        output_csv=args.output,
        model_path=args.model,
        device=args.device
    )

if __name__ == "__main__":
    main()
