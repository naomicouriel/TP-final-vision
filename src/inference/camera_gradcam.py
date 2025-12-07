"""
Real-time camera inference with Grad-CAM visualization.
"""
import cv2
import time
import numpy as np
from typing import Callable, Optional
import logging

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
        enable_gradcam: bool = True
    ):
        self.predictor = predictor
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.enable_gradcam = enable_gradcam
        
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
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to grab frame")
                    break
                
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
                
                # Draw results
                class_name = result.get('class_name', str(result['class_id']))
                conf = result['confidence']
                
                # Main prediction text
                text = f"{class_name}: {conf:.2f}"
                cv2.putText(display_frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Inference time
                time_text = f"{inference_time:.1f}ms"
                cv2.putText(display_frame, time_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show probabilities for all classes
                y_offset = 110
                for class_id, prob in enumerate(result['probabilities']):
                    class_label = self.predictor.class_mapping.get(class_id, f"Class {class_id}")
                    prob_text = f"{class_label}: {prob:.3f}"
                    color = (0, 255, 0) if class_id == result['class_id'] else (200, 200, 200)
                    cv2.putText(display_frame, prob_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 25
                
                # Grad-CAM status
                gradcam_status = "ON" if self.enable_gradcam else "OFF"
                cv2.putText(display_frame, f"Grad-CAM: {gradcam_status}", 
                           (display_frame.shape[1] - 180, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Instructions
                cv2.putText(display_frame, "Press 'q': quit | 'g': toggle Grad-CAM | 's': save", 
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
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
