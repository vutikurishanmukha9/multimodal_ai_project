"""
Multimodal AI System - Main integration module.

This module provides the MultimodalAI class that combines:
- Image processing with OpenCV
- Vision-language modeling with BLIP
- Comprehensive multimodal analysis pipeline
"""

import logging
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import json
import os
from datetime import datetime

from .image_processor import ImageProcessor
from .llm_integration import LLMProcessor
from .utils import load_image_pil, validate_image_format, create_output_directory

# Set up logging
logger = logging.getLogger(__name__)


class MultimodalAI:
    """
    Main multimodal AI system that combines computer vision and language understanding.

    This class orchestrates the entire pipeline:
    1. Image loading and preprocessing
    2. Feature extraction (objects, colors, text)
    3. Image captioning
    4. Question answering based on visual content
    """

    def __init__(self, 
                 yolo_model: str = "yolov8n.pt",
                 blip_model: str = "Salesforce/blip-image-captioning-base",
                 device: str = "auto"):
        """
        Initialize the multimodal AI system.

        Args:
            yolo_model (str): YOLO model for object detection
            blip_model (str): BLIP model for image captioning
            device (str): Computation device ('auto', 'cpu', 'cuda')
        """
        self.device = device
        self.yolo_model = yolo_model
        self.blip_model = blip_model

        # Initialize processors
        self.image_processor = None
        self.llm_processor = None

        # Initialize components
        self._initialize_components()

        # Store results from last processing
        self.last_results = {}

        logger.info("MultimodalAI system initialized successfully")

    def _initialize_components(self) -> None:
        """Initialize the image processor and LLM processor."""
        try:
            logger.info("Initializing image processor...")
            self.image_processor = ImageProcessor(yolo_model=self.yolo_model)

            logger.info("Initializing LLM processor...")
            self.llm_processor = LLMProcessor(
                model_id=self.blip_model,
                device=self.device
            )

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def process(self, image_path: str, question: str,
                analysis_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main processing pipeline for comprehensive multimodal analysis.

        Args:
            image_path (str): Path to the input image
            question (str): Question to answer about the image
            analysis_config (Optional[Dict[str, Any]]): Configuration for analysis steps

        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        if not validate_image_format(image_path):
            error_msg = f"Unsupported image format: {image_path}"
            logger.error(error_msg)
            return {'error': error_msg}

        if not os.path.exists(image_path):
            error_msg = f"Image file not found: {image_path}"
            logger.error(error_msg)
            return {'error': error_msg}

        # Use default config if none provided
        if analysis_config is None:
            analysis_config = self._get_default_config()

        logger.info(f"Starting multimodal analysis of: {image_path}")
        logger.info(f"Question: {question}")

        results = {
            'image_path': image_path,
            'question': question,
            'timestamp': datetime.now().isoformat(),
            'analysis_config': analysis_config,
            'processing_steps': {}
        }

        try:
            # Step 1: Load and preprocess image
            logger.info("Step 1: Image preprocessing")
            preprocessed = self.image_processor.preprocess(
                image_path,
                target_size=analysis_config.get('target_size', (640, 640)),
                normalize=analysis_config.get('normalize', True)
            )

            if preprocessed is None:
                error_msg = "Failed to preprocess image"
                logger.error(error_msg)
                results['error'] = error_msg
                return results

            results['processing_steps']['preprocessing'] = {
                'status': 'success',
                'image_shape': preprocessed.shape
            }

            # Step 2: Extract visual features
            logger.info("Step 2: Feature extraction")
            features = self._extract_features(analysis_config)
            results['features'] = features
            results['processing_steps']['feature_extraction'] = {
                'status': 'success',
                'features_extracted': list(features.keys())
            }

            # Step 3: Generate image caption
            logger.info("Step 3: Image captioning")
            caption_result = self._generate_caption(image_path, analysis_config)
            results['caption'] = caption_result
            results['processing_steps']['captioning'] = {
                'status': 'success' if not caption_result.get('error') else 'error',
                'caption_length': len(caption_result.get('caption', ''))
            }

            # Step 4: Answer question based on combined context
            logger.info("Step 4: Question answering")
            answer_result = self._answer_question(caption_result, features, question, analysis_config)
            results['answer'] = answer_result
            results['processing_steps']['question_answering'] = {
                'status': 'success' if not answer_result.get('error') else 'error',
                'answer_length': len(answer_result.get('answer', ''))
            }

            # Step 5: Generate comprehensive summary
            logger.info("Step 5: Generating summary")
            summary = self._generate_summary(results)
            results['summary'] = summary

            # Store results for future reference
            self.last_results = results

            logger.info("Multimodal analysis completed successfully")

        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            logger.error(error_msg)
            results['error'] = error_msg

        return results

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for analysis."""
        return {
            'target_size': (640, 640),
            'normalize': True,
            'object_detection': {
                'enabled': True,
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45
            },
            'color_extraction': {
                'enabled': True,
                'num_colors': 5,
                'method': 'kmeans'
            },
            'text_extraction': {
                'enabled': True,
                'preprocessing': {
                    'convert_to_gray': True,
                    'apply_gaussian_blur': True,
                    'apply_threshold': True
                }
            },
            'captioning': {
                'max_length': 50,
                'num_beams': 5,
                'temperature': 1.0,
                'conditional_text': None
            },
            'question_answering': {
                'max_length': 100,
                'num_beams': 5,
                'temperature': 1.0,
                'use_features_context': True
            }
        }

    def _extract_features(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract visual features from the image."""
        features = {}

        try:
            # Object detection
            if config.get('object_detection', {}).get('enabled', True):
                obj_config = config['object_detection']
                detections = self.image_processor.detect_objects(
                    confidence_threshold=obj_config.get('confidence_threshold', 0.5),
                    iou_threshold=obj_config.get('iou_threshold', 0.45)
                )
                features['objects'] = detections
                logger.info(f"Detected {len(detections)} objects")

            # Color extraction
            if config.get('color_extraction', {}).get('enabled', True):
                color_config = config['color_extraction']
                colors = self.image_processor.extract_colors(
                    num_colors=color_config.get('num_colors', 5),
                    method=color_config.get('method', 'kmeans')
                )
                features['colors'] = colors
                logger.info(f"Extracted {len(colors)} dominant colors")

            # Text extraction (OCR)
            if config.get('text_extraction', {}).get('enabled', True):
                text_config = config['text_extraction']
                ocr_result = self.image_processor.extract_text(
                    preprocessing_config=text_config.get('preprocessing')
                )
                features['ocr_text'] = ocr_result
                logger.info(f"Extracted text: '{ocr_result.get('text', '')[:50]}...'")

            # Image statistics
            features['image_stats'] = self.image_processor.get_image_statistics()

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            features['error'] = str(e)

        return features

    def _generate_caption(self, image_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image caption using BLIP."""
        try:
            caption_config = config.get('captioning', {})

            caption_result = self.llm_processor.caption_image(
                image=image_path,
                conditional_text=caption_config.get('conditional_text'),
                max_length=caption_config.get('max_length', 50),
                num_beams=caption_config.get('num_beams', 5),
                temperature=caption_config.get('temperature', 1.0)
            )

            logger.info(f"Generated caption: '{caption_result.get('caption', '')}'")
            return caption_result

        except Exception as e:
            error_msg = f"Error generating caption: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg, 'caption': ''}

    def _answer_question(self, caption_result: Dict[str, Any], 
                        features: Dict[str, Any], 
                        question: str,
                        config: Dict[str, Any]) -> Dict[str, Any]:
        """Answer question based on image caption and features."""
        try:
            qa_config = config.get('question_answering', {})

            # Build comprehensive context
            context_parts = []

            # Add caption
            caption = caption_result.get('caption', '')
            if caption:
                context_parts.append(f"Image description: {caption}")

            # Add features context if enabled
            if qa_config.get('use_features_context', True):
                # Add object information
                objects = features.get('objects', [])
                if objects:
                    object_names = [obj['class_name'] for obj in objects]
                    object_counts = {}
                    for name in object_names:
                        object_counts[name] = object_counts.get(name, 0) + 1

                    object_desc = ", ".join([f"{count} {name}{'s' if count > 1 else ''}" 
                                           for name, count in object_counts.items()])
                    context_parts.append(f"Objects detected: {object_desc}")

                # Add color information
                colors = features.get('colors', [])
                if colors and len(colors) > 0:
                    dominant_colors = [color['hex'] for color in colors[:3]]
                    context_parts.append(f"Dominant colors: {', '.join(dominant_colors)}")

                # Add OCR text if available
                ocr_text = features.get('ocr_text', {}).get('text', '').strip()
                if ocr_text:
                    context_parts.append(f"Text in image: {ocr_text}")

            # Combine context
            full_context = ". ".join(context_parts)

            # Generate answer
            answer_result = self.llm_processor.answer_question(
                caption=full_context,
                question=question,
                max_length=qa_config.get('max_length', 100),
                num_beams=qa_config.get('num_beams', 5),
                temperature=qa_config.get('temperature', 1.0)
            )

            # Add context information to result
            answer_result['context_used'] = full_context
            answer_result['features_included'] = qa_config.get('use_features_context', True)

            logger.info(f"Generated answer: '{answer_result.get('answer', '')}'")
            return answer_result

        except Exception as e:
            error_msg = f"Error answering question: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg, 'answer': ''}

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of the analysis."""
        try:
            summary = {
                'total_processing_time': 'N/A',  # Could be calculated if needed
                'analysis_success': not results.get('error'),
                'features_summary': {},
                'key_findings': []
            }

            # Summarize features
            features = results.get('features', {})

            if 'objects' in features:
                objects = features['objects']
                summary['features_summary']['objects_detected'] = len(objects)
                if objects:
                    unique_classes = list(set(obj['class_name'] for obj in objects))
                    summary['features_summary']['unique_object_classes'] = len(unique_classes)
                    summary['key_findings'].append(f"Detected {len(objects)} objects of {len(unique_classes)} different types")

            if 'colors' in features:
                colors = features['colors']
                summary['features_summary']['dominant_colors_extracted'] = len(colors)
                if colors:
                    top_color = colors[0]['hex'] if colors else None
                    summary['key_findings'].append(f"Most dominant color: {top_color}")

            if 'ocr_text' in features:
                ocr_result = features['ocr_text']
                text_found = bool(ocr_result.get('text', '').strip())
                summary['features_summary']['text_detected'] = text_found
                if text_found:
                    word_count = ocr_result.get('word_count', 0)
                    summary['key_findings'].append(f"Detected {word_count} words in image")

            # Add caption and answer info
            caption = results.get('caption', {}).get('caption', '')
            answer = results.get('answer', {}).get('answer', '')

            summary['caption_generated'] = bool(caption)
            summary['question_answered'] = bool(answer)

            if caption:
                summary['key_findings'].append(f"Generated caption: '{caption[:100]}...' " if len(caption) > 100 else f"Generated caption: '{caption}'")

            if answer:
                summary['key_findings'].append(f"Answer to question: '{answer[:100]}...' " if len(answer) > 100 else f"Answer to question: '{answer}'")

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {'error': str(e)}

    def save_results(self, results: Dict[str, Any], output_path: str,
                    include_images: bool = False) -> bool:
        """
        Save analysis results to file.

        Args:
            results (Dict[str, Any]): Analysis results to save
            output_path (str): Path to save results
            include_images (bool): Whether to save processed images

        Returns:
            bool: True if saved successfully
        """
        try:
            # Create output directory
            output_dir = os.path.dirname(output_path)
            if output_dir and not create_output_directory(output_dir):
                return False

            # Prepare results for JSON serialization
            json_results = self._prepare_for_json(results)

            # Save JSON results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)

            # Save images if requested
            if include_images and self.image_processor and self.image_processor.current_image is not None:
                base_path = os.path.splitext(output_path)[0]

                # Save original with detections
                features = results.get('features', {})
                objects = features.get('objects', [])

                if objects:
                    detection_image = self.image_processor.draw_detections(objects)
                    detection_path = f"{base_path}_detections.jpg"
                    cv2.imwrite(detection_path, detection_image)
                    logger.info(f"Saved detection image: {detection_path}")

            logger.info(f"Results saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False

    def _prepare_for_json(self, obj: Any) -> Any:
        """Prepare object for JSON serialization by handling numpy types."""
        import numpy as np

        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the multimodal system."""
        try:
            info = {
                'system_version': '1.0.0',
                'yolo_model': self.yolo_model,
                'blip_model': self.blip_model,
                'device': self.device,
                'components_loaded': {
                    'image_processor': self.image_processor is not None,
                    'llm_processor': self.llm_processor is not None and self.llm_processor.model_loaded
                }
            }

            # Add component-specific info
            if self.llm_processor and self.llm_processor.model_loaded:
                info['llm_info'] = self.llm_processor.get_model_info()

            return info

        except Exception as e:
            logger.error(f"Error getting system info: {str(e)}")
            return {'error': str(e)}

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.llm_processor:
                self.llm_processor.cleanup()

            # Clear stored results
            self.last_results = {}

            logger.info("MultimodalAI system cleaned up successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
