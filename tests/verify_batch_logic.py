import sys
import os
import numpy as np
from PIL import Image
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multimodal_system import MultimodalAI

def create_dummy_image(path):
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), -1)
    cv2.imwrite(path, img)
    return path

def test_batch():
    print("Initializing MultimodalAI...")
    ai = MultimodalAI() # Default BLIP
    
    img_paths = ["test_batch_1.jpg", "test_batch_2.jpg"]
    for p in img_paths:
        create_dummy_image(p)
        
    questions = ["What is this?", "Describe image."]
    
    print(f"Testing process_batch with {len(img_paths)} items...")
    
    try:
        results = ai.process_batch(img_paths, questions)
        
        print("\n--- Batch Results ---")
        for i, res in enumerate(results):
            if 'error' in res:
                print(f"Item {i} Failed: {res['error']}")
            else:
                print(f"Item {i} Success:")
                print(f"  Caption: {res['caption']['caption']}")
                print(f"  Objects: {len(res['features']['objects'])}")
                
        if len(results) == 2 and all('error' not in r for r in results):
            print("\nSUCCESS: Batch processing worked!")
        else:
            print("\nFAILURE: Batch processing had errors.")
            
    except Exception as e:
        print(f"Exception during batch processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for p in img_paths:
            if os.path.exists(p):
                os.remove(p)

if __name__ == "__main__":
    test_batch()
