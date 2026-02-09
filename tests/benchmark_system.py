import sys
import os
import time
import numpy as np
import cv2
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multimodal_system import MultimodalAI

def create_dummy_images(count=8):
    paths = []
    for i in range(count):
        path = f"bench_img_{i}.jpg"
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        cv2.imwrite(path, img)
        paths.append(path)
    return paths

def cleanup_images(paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)

def run_benchmark(model_type, batch_mode, image_paths, questions, ai_system):
    print(f"\n--- Benchmarking {model_type.upper()} ({'Batch' if batch_mode else 'Sequential'}) ---")
    
    # Switch model if needed
    # internally this handles unloading/loading
    if ai_system.llm_type != model_type:
        ai_system._switch_llm_service(model_type)
        
    start_time = time.time()
    
    if batch_mode:
        results = ai_system.process_batch(image_paths, questions)
    else:
        results = []
        for i, path in enumerate(image_paths):
            res = ai_system.process(path, questions[i])
            results.append(res)
            
    end_time = time.time()
    duration = end_time - start_time
    fps = len(image_paths) / duration
    
    print(f"Processed {len(image_paths)} images in {duration:.2f}s")
    print(f"FPS: {fps:.2f}")
    return fps, duration

def main():
    print("Initializing System for Benchmark...")
    # Initialize with BLIP first
    ai = MultimodalAI(llm_type="blip")
    
    num_images = 8 # Enough to see batch benefits
    image_paths = create_dummy_images(num_images)
    questions = ["Describe this image."] * num_images
    
    results = {}
    
    try:
        # 1. BLIP Sequential
        fps, dur = run_benchmark("blip", False, image_paths, questions, ai)
        results['blip_seq'] = fps
        
        # 2. BLIP Batch
        fps, dur = run_benchmark("blip", True, image_paths, questions, ai)
        results['blip_batch'] = fps
        
        # 3. LlaVA Benchmarks
        # Note: LlaVA loading takes time, exclude from measurement logic inside the function but wait for it
        print("\nSwitching to LlaVA for benchmarks (this may take a moment)...")
        
        # 3. LlaVA Sequential
        fps, dur = run_benchmark("llava", False, image_paths, questions, ai)
        results['llava_seq'] = fps
        
        # 4. LlaVA Batch
        fps, dur = run_benchmark("llava", True, image_paths, questions, ai)
        results['llava_batch'] = fps
        
        print("\n" + "="*40)
        print("FINAL RESULTS (FPS - Higher is Better)")
        print("="*40)
        print(f"BLIP Sequential: {results['blip_seq']:.2f}")
        print(f"BLIP Batch     : {results['blip_batch']:.2f}  (x{results['blip_batch']/results['blip_seq']:.1f} speedup)")
        print("-" * 40)
        print(f"LlaVA Sequential: {results['llava_seq']:.2f}")
        print(f"LlaVA Batch     : {results['llava_batch']:.2f}  (x{results['llava_batch']/results['llava_seq']:.1f} speedup)")
        print("="*40)
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_images(image_paths)

if __name__ == "__main__":
    main()
