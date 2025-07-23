# Create utils.py - shared helper functions
utils_content = '''"""
Utility functions for the multimodal AI system.

This module provides shared helper functions for image loading, preprocessing,
and common operations used across the multimodal AI pipeline.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file path using OpenCV.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Optional[np.ndarray]: Loaded image as numpy array in BGR format, None if failed
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
            
        logger.info(f"Successfully loaded image: {image_path} with shape {image.shape}")
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def load_image_pil(image_path: str) -> Optional[Image.Image]:
    """
    Load an image using PIL/Pillow for compatibility with transformers.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Optional[Image.Image]: PIL Image object, None if failed
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        image = Image.open(image_path).convert('RGB')
        logger.info(f"Successfully loaded PIL image: {image_path} with size {image.size}")
        return image
        
    except Exception as e:
        logger.error(f"Error loading PIL image {image_path}: {str(e)}")
        return None


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target dimensions.
    
    Args:
        image (np.ndarray): Input image
        target_size (Tuple[int, int]): Target (width, height)
        maintain_aspect (bool): Whether to maintain aspect ratio
        
    Returns:
        np.ndarray: Resized image
    """
    try:
        if maintain_aspect:
            # Calculate scaling factor to fit within target size
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create padded image if needed
            if new_w != target_w or new_h != target_h:
                # Create black background
                padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
                
                # Center the resized image
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
                
                return padded
            return resized
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Normalized image
    """
    try:
        return image.astype(np.float32) / 255.0
    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        return image


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB format.
    
    Args:
        image (np.ndarray): BGR image
        
    Returns:
        np.ndarray: RGB image
    """
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error converting BGR to RGB: {str(e)}")
        return image


def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR format.
    
    Args:
        image (np.ndarray): RGB image
        
    Returns:
        np.ndarray: BGR image
    """
    try:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error converting RGB to BGR: {str(e)}")
        return image


def create_output_directory(output_dir: str) -> bool:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir (str): Directory path to create
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory ready: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {str(e)}")
        return False


def validate_image_format(image_path: str) -> bool:
    """
    Validate if file is a supported image format.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        bool: True if supported format
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    try:
        _, ext = os.path.splitext(image_path.lower())
        return ext in supported_formats
    except Exception as e:
        logger.error(f"Error validating image format: {str(e)}")
        return False


def draw_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                     label: str = "", color: Tuple[int, int, int] = (0, 255, 0), 
                     thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box with label on image.
    
    Args:
        image (np.ndarray): Input image
        bbox (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2)
        label (str): Text label to display
        color (Tuple[int, int, int]): BGR color for box and text
        thickness (int): Line thickness
        
    Returns:
        np.ndarray: Image with bounding box drawn
    """
    try:
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if label:
            # Get text size for background rectangle
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(image, (x1, y1 - text_h - baseline - 5), 
                         (x1 + text_w, y1), color, -1)
            
            # Draw text
            cv2.putText(image, label, (x1, y1 - baseline - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return image
        
    except Exception as e:
        logger.error(f"Error drawing bounding box: {str(e)}")
        return image


def get_image_info(image: np.ndarray) -> dict:
    """
    Get basic information about an image.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        dict: Image information including shape, dtype, etc.
    """
    try:
        info = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min_value': float(np.min(image)),
            'max_value': float(np.max(image)),
            'mean_value': float(np.mean(image))
        }
        
        if len(image.shape) == 3:
            info['channels'] = image.shape[2]
        else:
            info['channels'] = 1
            
        return info
        
    except Exception as e:
        logger.error(f"Error getting image info: {str(e)}")
        return {}
'''

with open("multimodal_ai_project/src/utils.py", "w") as f:
    f.write(utils_content)

print("Created utils.py with comprehensive helper functions")