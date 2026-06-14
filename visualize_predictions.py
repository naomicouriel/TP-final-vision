"""
Visualize prediction results - create a summary showing correctly and incorrectly classified images.
Useful for understanding what the model struggles with.
"""
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def create_visualization_grid(csv_path: str, output_path: str = 'prediction_visualization.png'):
    """
    Create a grid visualization of predictions.
    Shows examples of correct and incorrect predictions.
    """
    # Load predictions
    df = pd.read_csv(csv_path)
    
    # Separate correct and incorrect
    correct_df = df[df['predicted_class'] == 'ecoglasses'].sort_values('confidence', ascending=False)
    incorrect_df = df[df['predicted_class'] != 'ecoglasses'].sort_values('confidence', ascending=False)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Title
    fig.suptitle(f'Ecoglasses Prediction Results: {len(correct_df)}/{len(df)} Correct ({len(correct_df)/len(df)*100:.1f}%)', 
                 fontsize=20, fontweight='bold')
    
    # Top row: Best correct predictions
    ax_correct = fig.add_subplot(2, 1, 1)
    ax_correct.set_title('✅ Correctly Classified as Ecoglasses (Top 10 by confidence)', 
                         fontsize=16, fontweight='bold', color='green')
    ax_correct.axis('off')
    
    correct_images = []
    for i, (_, row) in enumerate(correct_df.head(10).iterrows()):
        img = cv2.imread(row['filepath'])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (150, 150))
            
            # Add confidence text
            cv2.putText(img, f"{row['confidence']:.3f}", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            correct_images.append(img)
    
    if correct_images:
        correct_grid = np.hstack(correct_images)
        ax_correct.imshow(correct_grid)
    
    # Bottom row: Misclassified (worst predictions)
    ax_incorrect = fig.add_subplot(2, 1, 2)
    ax_incorrect.set_title('❌ Misclassified Images (Top 10 by confidence in wrong class)', 
                          fontsize=16, fontweight='bold', color='red')
    ax_incorrect.axis('off')
    
    incorrect_images = []
    incorrect_labels = []
    for i, (_, row) in enumerate(incorrect_df.head(10).iterrows()):
        img = cv2.imread(row['filepath'])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (150, 150))
            
            # Add prediction text
            pred_class = row['predicted_class'].replace('_', '\n')
            cv2.putText(img, f"{pred_class}", (5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(img, f"{row['confidence']:.3f}", (5, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            incorrect_images.append(img)
            incorrect_labels.append(row['predicted_class'])
    
    if incorrect_images:
        incorrect_grid = np.hstack(incorrect_images)
        ax_incorrect.imshow(incorrect_grid)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Correctly classified: {len(correct_df)}/{len(df)} ({len(correct_df)/len(df)*100:.1f}%)")
    print(f"  Misclassified: {len(incorrect_df)}/{len(df)} ({len(incorrect_df)/len(df)*100:.1f}%)")
    
    print("\nMisclassification breakdown:")
    for pred_class, count in incorrect_df['predicted_class'].value_counts().items():
        print(f"  → {pred_class}: {count} images")
    
    return fig

def create_confusion_analysis(csv_path: str):
    """
    Analyze which classes ecoglasses are confused with.
    """
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*60)
    print("CONFUSION ANALYSIS")
    print("="*60)
    
    # For each image, find the top 2 predicted classes
    df['second_best_class'] = df[['prob_cardboard_paper', 'prob_ecoglasses', 
                                   'prob_metal_plastic', 'prob_trash']].apply(
        lambda row: sorted([
            ('cardboard_paper', row['prob_cardboard_paper']),
            ('ecoglasses', row['prob_ecoglasses']),
            ('metal_plastic', row['prob_metal_plastic']),
            ('trash', row['prob_trash'])
        ], key=lambda x: x[1], reverse=True)[1][0], axis=1
    )
    
    df['second_best_prob'] = df[['prob_cardboard_paper', 'prob_ecoglasses', 
                                  'prob_metal_plastic', 'prob_trash']].apply(
        lambda row: sorted([
            row['prob_cardboard_paper'],
            row['prob_ecoglasses'],
            row['prob_metal_plastic'],
            row['prob_trash']
        ], reverse=True)[1], axis=1
    )
    
    # Analyze close calls (where second best is close to best)
    close_calls = df[abs(df['confidence'] - df['second_best_prob']) < 0.2]
    
    print(f"\nClose calls (top 2 classes within 0.2 probability): {len(close_calls)}")
    print("\nMost common confusions:")
    for _, row in close_calls.head(10).iterrows():
        print(f"  {Path(row['filename']).name}: {row['predicted_class']} ({row['confidence']:.3f}) "
              f"vs {row['second_best_class']} ({row['second_best_prob']:.3f})")
    
    # Average probabilities by prediction
    print("\n\nAverage class probabilities by prediction:")
    print("-"*60)
    for pred_class in df['predicted_class'].unique():
        subset = df[df['predicted_class'] == pred_class]
        print(f"\nWhen predicted as '{pred_class}' ({len(subset)} images):")
        print(f"  cardboard_paper: {subset['prob_cardboard_paper'].mean():.3f}")
        print(f"  ecoglasses:      {subset['prob_ecoglasses'].mean():.3f}")
        print(f"  metal_plastic:   {subset['prob_metal_plastic'].mean():.3f}")
        print(f"  trash:           {subset['prob_trash'].mean():.3f}")

def main():
    csv_path = 'ecoglasses_bandeja_predictions.csv'
    
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found!")
        print("Run: python batch_predict.py data/custom/ecoglasses_bandeja_jpg")
        return
    
    print("Creating visualization...")
    create_visualization_grid(csv_path)
    
    print("\nAnalyzing confusions...")
    create_confusion_analysis(csv_path)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("  - Visualization: prediction_visualization.png")
    print("  - Full results: ecoglasses_bandeja_predictions.csv")
    print("="*60)

if __name__ == "__main__":
    main()
