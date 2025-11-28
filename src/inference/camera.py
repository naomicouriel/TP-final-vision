"""
Real-time camera inference module.
"""
import cv2
import time
import numpy as np
from typing import Callable, Optional
import logging

from src.inference.predictor import RecyclingPredictor

logger = logging.getLogger(__name__)

class CameraInference:
    """
    Class to handle real-time inference from camera feed.
    """
    def __init__(
        self,
        predictor: RecyclingPredictor,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        self.predictor = predictor
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False

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
        Run inference loop.
        
        Args:
            callback: Optional function to call with prediction result (e.g. trigger servomotor)
        """
        if not self.is_running:
            self.start()
            
        logger.info("Starting inference loop. Press 'q' to quit.")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to grab frame")
                    break
                
                # Preprocess for display (keep BGR)
                display_frame = frame.copy()
                
                # Run inference (convert to RGB inside predictor)
                # We pass the BGR frame directly, predictor handles conversion if numpy array
                # Note: predictor.preprocess expects RGB if numpy, so we should convert here
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                start_time = time.time()
                result = self.predictor.predict(rgb_frame)
                inference_time = (time.time() - start_time) * 1000
                
                # Draw results
                class_name = result.get('class_name', str(result['class_id']))
                conf = result['confidence']
                
                text = f"{class_name}: {conf:.2f} ({inference_time:.1f}ms)"
                cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Recycling Classification', display_frame)
                
                # Trigger callback
                if callback:
                    callback(result)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

# Example servomotor callback placeholder
def trigger_servomotor(result: dict):
    """
    Placeholder for servomotor control logic.
    """
    cls = result.get('class_name')
    conf = result.get('confidence')
    
    if conf > 0.8:
        # Logic to send signal to Arduino/Raspberry Pi
        # print(f"Triggering bin for {cls}")
        pass
