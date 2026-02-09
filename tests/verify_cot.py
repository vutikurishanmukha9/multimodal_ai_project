import sys
import os
import numpy as np
from PIL import Image
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multimodal_system import MultimodalAI

def create_dummy_image(path):
    # Create a 640x640 rgb image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Draw a rectangle (fake object)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), -1)
    cv2.imwrite(path, img)
    return path

def test_cot():
    print("Initializing MultimodalAI...")
    ai = MultimodalAI()
    
    img_path = "test_image.jpg"
    create_dummy_image(img_path)
    
    print("Testing processing with Detailed Mode...")
    config = {
        'reasoning_mode': 'detailed',
        'object_detection': {'enabled': True, 'confidence_threshold': 0.1} # Low thresh to ensure detection on dummy
    }
    
    try:
        result = ai.process(img_path, "What is in this image?", analysis_config=config)
        
        print("\n--- Results ---")
        if 'error' in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
            
        print("Steps executed:")
        for step, info in result.get('processing_steps', {}).items():
            print(f"- {step}: {info['status']}")
            
        if 'visual_cot' in result['processing_steps']:
            print("SUCCESS: Visual CoT step was executed.")
        else:
            print("FAILURE: Visual CoT step was NOT executed.")
            
    except Exception as e:
        print(f"Exception during processing: {e}")
        raise
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    test_cot()
