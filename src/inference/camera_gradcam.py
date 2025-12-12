"""
Real-time camera inference with Grad-CAM visualization.
"""
import cv2
import time
import numpy as np
from typing import Callable, Optional
import logging
from collections import deque

from src.inference.predictor import RecyclingPredictor
from src.inference.gradcam import GradCAMPredictor

logger = logging.getLogger(__name__)


class CameraInferenceWithGradCAM:
    """
    Class to handle real-time inference from camera feed with Grad-CAM visualization.
    """
    def __init__(
        self,
        predictor: RecyclingPredictor,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_gradcam: bool = True,
        stability_duration: float = 4.0,  # seconds to wait for stable prediction
        stereo_mode: str = None  # Options: None, 'left', 'right'
    ):
        self.predictor = predictor
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.enable_gradcam = enable_gradcam
        
        # Stereo camera handling
        self.stereo_mode = stereo_mode  # None, 'left', or 'right'
        
        # Stability tracking for Arduino
        self.stability_duration = stability_duration
        self.prediction_history = deque(maxlen=100)  # Keep last 100 predictions
        self.last_stable_class = None
        self.last_sent_class = None
        
        # Zoom functionality
        self.zoom_level = 1.0
        self.zoom_step = 0.1
        self.min_zoom = 1.0
        self.max_zoom = 3.0
        
        # Setup Grad-CAM
        if self.enable_gradcam:
            self.gradcam_predictor = GradCAMPredictor(predictor)

    def start(self):
        """Start camera stream."""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
            
        self.is_running = True
        logger.info(f"Camera {self.camera_id} started.")

    def stop(self):
        """Stop camera stream."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera stopped.")

    def extract_stereo_view(self, frame):
        """Extract single view from stereo camera if stereo_mode is set."""
        if self.stereo_mode is None:
            return frame
        
        height, width = frame.shape[:2]
        half_width = width // 2
        
        if self.stereo_mode == 'left':
            # Use left half of the image
            return frame[:, :half_width]
        elif self.stereo_mode == 'right':
            # Use right half of the image
            return frame[:, half_width:]
        
        return frame
    
    def apply_zoom(self, frame):
        """Apply digital zoom to frame."""
        if self.zoom_level == 1.0:
            return frame
        
        height, width = frame.shape[:2]
        
        # Calculate crop size
        crop_width = int(width / self.zoom_level)
        crop_height = int(height / self.zoom_level)
        
        # Calculate center crop
        x = (width - crop_width) // 2
        y = (height - crop_height) // 2
        
        # Crop and resize back to original size
        cropped = frame[y:y+crop_height, x:x+crop_width]
        zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        
        return zoomed
    
    def check_stability(self, class_id: int, confidence: float, min_confidence: float = 0.7):
        """
        Check if the prediction has been stable for the required duration.
        
        Args:
            class_id: Current predicted class
            confidence: Confidence of prediction
            min_confidence: Minimum confidence to consider prediction valid
            
        Returns:
            tuple: (is_stable, stable_class_id) - Whether prediction is stable and which class
        """
        current_time = time.time()
        
        # Only track high-confidence predictions
        if confidence >= min_confidence:
            self.prediction_history.append((current_time, class_id))
        
        # Remove old predictions outside stability window
        cutoff_time = current_time - self.stability_duration
        while self.prediction_history and self.prediction_history[0][0] < cutoff_time:
            self.prediction_history.popleft()
        
        # Check if we have enough predictions in the window
        if len(self.prediction_history) < 5:  # Need at least 5 predictions
            return False, None
        
        # Check if all recent predictions are the same class
        recent_classes = [cls for _, cls in self.prediction_history]
        if len(set(recent_classes)) == 1:  # All predictions are the same
            stable_class = recent_classes[0]
            
            # Only return True if this is different from last sent class
            if stable_class != self.last_sent_class:
                self.last_stable_class = stable_class
                return True, stable_class
        
        return False, None
    
    def process_frame(self, result):
        """
        Process a single frame and return prediction result.
        Can be overridden by subclasses (e.g., for Arduino integration).
        """
        # This method can be overridden for custom processing
        return None

    def run(self, callback: Optional[Callable[[dict], None]] = None):
        """
        Run inference loop with Grad-CAM visualization.
        
        Args:
            callback: Optional function to call with prediction result
        """
        if not self.is_running:
            self.start()
            
        logger.info("Starting inference loop with Grad-CAM visualization.")
        logger.info("Press 'q' to quit, 'g' to toggle Grad-CAM, 's' to save frame")
        logger.info("Press '+'/'-' to zoom in/out")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to grab frame")
                    break
                
                # Extract single view if stereo camera
                frame = self.extract_stereo_view(frame)
                
                # Apply zoom
                frame = self.apply_zoom(frame)
                
                # Convert to RGB for model
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                start_time = time.time()
                
                if self.enable_gradcam:
                    # Run inference with Grad-CAM
                    result, gradcam_overlay = self.gradcam_predictor.predict_with_cam(rgb_frame)
                    # Convert overlay back to BGR for display
                    display_frame = cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR)
                else:
                    # Run normal inference
                    result = self.predictor.predict(rgb_frame)
                    display_frame = frame.copy()
                
                inference_time = (time.time() - start_time) * 1000
                
                # Check stability and call process_frame for custom behavior
                class_id = result['class_id']
                confidence = result['confidence']
                is_stable, stable_class = self.check_stability(class_id, confidence)
                
                # Call custom process_frame (for Arduino integration)
                if is_stable and stable_class is not None:
                    result['stable_class'] = stable_class
                    self.process_frame(result)
                
                # Draw results
                class_name = result.get('class_name', str(result['class_id']))
                conf = result['confidence']
                
                # Main prediction text with stability indicator
                stability_indicator = "✓ STABLE" if is_stable else ""
                text = f"{class_name}: {conf:.2f} {stability_indicator}"
                color = (0, 255, 0) if is_stable else (255, 255, 0)
                cv2.putText(display_frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Inference time
                time_text = f"{inference_time:.1f}ms"
                cv2.putText(display_frame, time_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show probabilities for all classes
                y_offset = 110
                for idx, prob in enumerate(result['probabilities']):
                    class_label = self.predictor.class_mapping.get(idx, f"Class {idx}")
                    prob_text = f"{class_label}: {prob:.3f}"
                    color = (0, 255, 0) if idx == result['class_id'] else (200, 200, 200)
                    cv2.putText(display_frame, prob_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 25
                
                # Grad-CAM status
                gradcam_status = "ON" if self.enable_gradcam else "OFF"
                cv2.putText(display_frame, f"Grad-CAM: {gradcam_status}", 
                           (display_frame.shape[1] - 220, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Zoom level and stereo mode
                zoom_text = f"Zoom: {self.zoom_level:.1f}x"
                if self.stereo_mode:
                    zoom_text += f" ({self.stereo_mode.upper()})"
                cv2.putText(display_frame, zoom_text, 
                           (display_frame.shape[1] - 220, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Instructions
                cv2.putText(display_frame, "q: quit | g: Grad-CAM | s: save | +/-: zoom", 
                           (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Recycling Classification with Grad-CAM', display_frame)
                
                # Trigger callback
                if callback:
                    callback(result)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('g'):
                    self.enable_gradcam = not self.enable_gradcam
                    status = "enabled" if self.enable_gradcam else "disabled"
                    print(f"Grad-CAM {status}")
                elif key == ord('s'):
                    filename = f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved frame to {filename}")
                elif key == ord('+') or key == ord('='):
                    self.zoom_level = min(self.zoom_level + self.zoom_step, self.max_zoom)
                    print(f"Zoom: {self.zoom_level:.1f}x")
                elif key == ord('-') or key == ord('_'):
                    self.zoom_level = max(self.zoom_level - self.zoom_step, self.min_zoom)
                    print(f"Zoom: {self.zoom_level:.1f}x")
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
