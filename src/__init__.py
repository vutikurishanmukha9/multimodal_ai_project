"""
Multimodal AI System package.

This package provides a comprehensive multimodal AI system that combines 
computer vision and natural language processing for image analysis.

Modules:
- image_processor: OpenCV-based image processing and feature extraction
- llm_integration: BLIP vision-language model integration
- multimodal_system: Main system orchestrator combining all components
- utils: Shared utility functions
- web_app: Streamlit-based web interface

Example usage:
    from src.multimodal_system import MultimodalAI
    
    system = MultimodalAI()
    results = system.process("image.jpg", "What do you see?")
    print(results['answer'])
"""

from .image_processor import ImageProcessor
from .llm_integration import LLMProcessor
from .multimodal_system import MultimodalAI
from .utils import (
    load_image,
    load_image_pil,
    resize_image,
    normalize_image,
    convert_bgr_to_rgb,
    convert_rgb_to_bgr,
    validate_image_format,
    draw_bounding_box,
    get_image_info
)

__version__ = "1.0.0"
__author__ = "Multimodal AI Team"

__all__ = [
    "ImageProcessor",
    "LLMProcessor", 
    "MultimodalAI",
    "load_image",
    "load_image_pil",
    "resize_image",
    "normalize_image",
    "convert_bgr_to_rgb",
    "convert_rgb_to_bgr",
    "validate_image_format",
    "draw_bounding_box",
    "get_image_info"
]
