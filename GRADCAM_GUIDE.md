# Grad-CAM Visualization Guide

## What is Grad-CAM?

**Grad-CAM** (Gradient-weighted Class Activation Mapping) is a visualization technique that shows which regions of an input image are most important for the neural network's prediction.

### How it Works:
1. Model makes a prediction on an image
2. Gradients flow backward from the predicted class to the last convolutional layer
3. These gradients are weighted by importance and combined with activations
4. Result: A heatmap showing which pixels influenced the decision

### Color Interpretation:
- 🔴 **Red/Yellow regions**: High importance - the model focuses on these areas
- 🟠 **Orange regions**: Medium importance
- 🔵 **Blue/Purple regions**: Low importance - these areas don't affect the prediction

---

## Using Grad-CAM in Your Project

### 1. Static Image Visualization

```python
from src.inference.gradcam import GradCAMPredictor
import cv2

# Initialize
gradcam_predictor = GradCAMPredictor(predictor)

# Load and predict with Grad-CAM
image = cv2.imread('path/to/image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result, gradcam_overlay = gradcam_predictor.predict_with_cam(image_rgb)

# Display
import matplotlib.pyplot as plt
plt.imshow(gradcam_overlay)
plt.title(f"{result['class_name']}: {result['confidence']:.2f}")
plt.axis('off')
plt.show()
```

### 2. Real-time Camera with Grad-CAM

```python
from src.inference.camera_gradcam import CameraInferenceWithGradCAM

# Start camera with Grad-CAM enabled
camera = CameraInferenceWithGradCAM(
    predictor, 
    camera_id=0,
    enable_gradcam=True
)

camera.run()
```

**Interactive Controls:**
- **'q'**: Quit the camera
- **'g'**: Toggle Grad-CAM on/off (useful for comparing with/without heatmap)
- **'s'**: Save the current frame (with timestamp filename)

### 3. Using in Jupyter Notebook

See `notebooks/03_inference_demo.ipynb`:
- **Cell 2.5**: Static image Grad-CAM visualization (4 sample images)
- **Cell 3**: Real-time camera with Grad-CAM

---

## Understanding Grad-CAM Output

### What Grad-CAM Shows:

#### ✅ **Good Predictions**
When the model correctly classifies an image, Grad-CAM typically highlights:
- **Ecoglasses**: Focus on the container body, lid, or distinctive features
- **Metal/Plastic**: Focus on reflective surfaces, material texture
- **Cardboard/Paper**: Focus on paper texture, cardboard edges
- **Trash**: Focus on mixed materials or non-recyclable features

#### ❌ **Poor Predictions**
When the model misclassifies:
- Heatmap may focus on irrelevant regions (background, shadows)
- Multiple scattered hot spots (uncertainty)
- Focus on confusing features (e.g., metallic ecoglasses frames → classified as metal)

### Example Interpretations:

**Case 1: Correct Ecoglasses Classification**
```
Heatmap focuses on: Glass body, distinctive shape
Prediction: ecoglasses (confidence: 0.85)
Interpretation: ✅ Model correctly identifies container features
```

**Case 2: Misclassified as Metal/Plastic**
```
Heatmap focuses on: Metallic rim, reflective surface
Prediction: metal_plastic (confidence: 0.72)
Interpretation: ❌ Model confused by reflective metallic frame
```

---

## Technical Details

### Implementation

**Location**: `src/inference/gradcam.py`

**Key Components:**
1. **GradCAM class**: Core implementation
   - Captures gradients during backward pass
   - Computes weighted activation maps
   - Generates normalized heatmap

2. **GradCAMPredictor class**: Wrapper for easy use
   - Integrates with existing RecyclingPredictor
   - Handles preprocessing and overlay generation

3. **Target Layer**: Last convolutional layer of MobileNetV3
   - `model.model.features[-1]`
   - Provides spatial information about important regions

### Performance Impact

- **Computational Overhead**: ~20-30% slower than regular inference
  - Regular inference: 15-30ms
  - With Grad-CAM: 20-40ms
- **Memory**: Requires storing gradients (minimal impact)
- **Toggle**: Press 'g' to disable Grad-CAM for faster inference

### Customization

#### Change Heatmap Colors
```python
# In gradcam.py, modify overlay_heatmap method
heatmap_colored = cv2.applyColorMap(
    np.uint8(255 * heatmap_resized), 
    cv2.COLORMAP_JET  # Try: COLORMAP_HOT, COLORMAP_TURBO, COLORMAP_VIRIDIS
)
```

#### Adjust Overlay Transparency
```python
# More transparent (see more of original image)
overlayed = cv2.addWeighted(image, 0.7, heatmap_colored, 0.3, 0)

# More opaque (emphasize heatmap)
overlayed = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)
```

#### Target Different Layer
```python
# In gradcam.py, get_target_layer_mobilenet function
# Try earlier layers for more localized features
target_layer = model.model.features[-3]  # 3rd from last layer
```

---

## Use Cases

### 1. Model Debugging
- **Identify why model makes mistakes**
- See if model focuses on correct features or background
- Detect dataset biases (e.g., model learns background instead of object)

### 2. Trust and Explainability
- **Show users why the model made a decision**
- Build confidence in automated recycling systems
- Useful for auditing and compliance

### 3. Data Collection Guidance
- **See what features matter most**
- Guide photographers on what to capture
- Identify missing features in training data

### 4. Model Improvement
- **Discover confusing visual patterns**
- Example: Ecoglasses' metallic frames confuse model
- Solution: Add more diverse ecoglasses with various frame types

---

## Troubleshooting

### Issue: Grad-CAM shows only blue (no hot spots)

**Cause**: Low gradients, model is very uncertain

**Solution**: 
- Check model confidence (might be predicting randomly)
- Ensure model is loaded correctly
- Verify image preprocessing

### Issue: Grad-CAM focuses on background

**Cause**: Model trained with background bias

**Solution**: 
- Add more diverse backgrounds to training data
- Use better data augmentation
- Apply background subtraction preprocessing

### Issue: Slow performance

**Cause**: Grad-CAM requires backward pass

**Solution**: 
- Press 'g' to toggle off when not needed
- Reduce camera resolution
- Use GPU for acceleration

### Issue: Heatmap doesn't align with image

**Cause**: Resizing mismatch

**Solution**: Check that heatmap is properly resized to image dimensions (handled automatically in current implementation)

---

## Advanced Usage

### Batch Processing with Grad-CAM

```python
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

gradcam_predictor = GradCAMPredictor(predictor)
image_dir = Path('data/custom/ecoglasses_bandeja_jpg')

for img_path in image_dir.glob('*.jpg'):
    image = cv2.imread(str(img_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    result, overlay = gradcam_predictor.predict_with_cam(image_rgb)
    
    # Save with Grad-CAM overlay
    output_path = f"gradcam_output/{img_path.stem}_gradcam.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print(f"{img_path.name}: {result['class_name']} ({result['confidence']:.3f})")
```

### Compare Multiple Models

```python
# Load two different models
predictor1 = RecyclingPredictor(model_path='outputs/checkpoints/best_model.pt', ...)
predictor2 = RecyclingPredictor(model_path='outputs/checkpoints/best_model_finetuned.pt', ...)

gradcam1 = GradCAMPredictor(predictor1)
gradcam2 = GradCAMPredictor(predictor2)

# Compare focus areas
result1, overlay1 = gradcam1.predict_with_cam(image)
result2, overlay2 = gradcam2.predict_with_cam(image)

# Visualize side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(overlay1)
axes[0].set_title(f"Original Model: {result1['class_name']}")
axes[1].imshow(overlay2)
axes[1].set_title(f"Fine-tuned Model: {result2['class_name']}")
plt.show()
```

---

## References

- **Original Paper**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
- **PyTorch Implementation**: Based on official Grad-CAM implementation patterns
- **MobileNetV3 Adaptation**: Optimized for efficient mobile architectures

---

## Summary

✅ **Grad-CAM helps you:**
- Understand what the model "sees"
- Debug classification errors
- Build trust in predictions
- Improve data collection

✅ **Easy to use:**
- Static images: 3 lines of code
- Real-time camera: Built-in toggle with 'g' key
- No model retraining needed

✅ **Practical applications:**
- Model debugging and improvement
- User trust and transparency
- Dataset quality assessment
- Feature importance analysis
