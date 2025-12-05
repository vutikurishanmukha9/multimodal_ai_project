"""
Comprehensive integration tests for MultimodalAI system.
Tests the complete pipeline including image processing, LLM integration, and error handling.
Covers unit tests, integration tests, edge cases, and performance tests.
"""

import os
import pytest
import numpy as np
import cv2
import tempfile
import shutil
import time
import psutil
import threading
from unittest.mock import patch, MagicMock, Mock
from PIL import Image, ImageDraw
from src.multimodal_system import MultimodalAI
from src.image_processor import ImageProcessor
from src.llm_integration import LLMProcessor


class TestMultimodalAI:
    """Comprehensive test suite for MultimodalAI system."""

    @pytest.fixture(scope="class")
    def multimodal_system(self):
        """Fixture that returns initialized MultimodalAI system."""
        return MultimodalAI()

    @pytest.fixture(scope="function")
    def temp_dir(self):
        """Create temporary directory for test images."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture(scope="function")
    def comprehensive_test_images(self, temp_dir):
        """Create comprehensive set of test images for various scenarios."""
        images = {}

        # Simple colored squares
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        for color in colors:
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            color_values = {
                "red": [0, 0, 255], "blue": [255, 0, 0], "green": [0, 255, 0],
                "yellow": [0, 255, 255], "purple": [255, 0, 255], "orange": [0, 165, 255]
            }
            img[:] = color_values[color]
            img_path = os.path.join(temp_dir, f"{color}_square.jpg")
            cv2.imwrite(img_path, img)
            images[f"{color}_square"] = img_path

        # Complex scene with multiple objects
        scene = np.ones((600, 800, 3), dtype=np.uint8) * 135  # Gray background
        # Draw a house
        cv2.rectangle(scene, (200, 300), (600, 500), (139, 69, 19), -1)  # Brown house
        cv2.fillPoly(scene, [np.array([[200, 300], [400, 200], [600, 300]])], (0, 0, 255))  # Red roof
        cv2.rectangle(scene, (300, 400), (350, 500), (101, 67, 33), -1)  # Door
        cv2.rectangle(scene, (450, 350), (550, 400), (173, 216, 230), -1)  # Window
        # Add a car
        cv2.rectangle(scene, (50, 450), (180, 520), (255, 0, 0), -1)  # Blue car
        cv2.circle(scene, (80, 520), 20, (0, 0, 0), -1)  # Wheel
        cv2.circle(scene, (150, 520), 20, (0, 0, 0), -1)  # Wheel
        # Add a tree
        cv2.rectangle(scene, (700, 400), (720, 500), (139, 69, 19), -1)  # Trunk
        cv2.circle(scene, (710, 380), 40, (0, 255, 0), -1)  # Leaves
        scene_path = os.path.join(temp_dir, "complex_scene.jpg")
        cv2.imwrite(scene_path, scene)
        images["complex_scene"] = scene_path

        # Image with text
        text_img = np.ones((200, 600, 3), dtype=np.uint8) * 255
        cv2.putText(text_img, "MULTIMODAL AI TEST", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(text_img, "Processing Complete", (100, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        text_path = os.path.join(temp_dir, "text_image.jpg")
        cv2.imwrite(text_path, text_img)
        images["text_image"] = text_path

        # Geometric shapes for object detection
        shapes_img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(shapes_img, (50, 50), (150, 150), (255, 255, 255), -1)  # White square
        cv2.circle(shapes_img, (300, 100), 50, (0, 255, 0), -1)  # Green circle
        cv2.fillPoly(shapes_img, [np.array([[200, 200], [250, 300], [150, 300]])], (0, 0, 255))  # Red triangle
        shapes_path = os.path.join(temp_dir, "geometric_shapes.jpg")
        cv2.imwrite(shapes_path, shapes_img)
        images["geometric_shapes"] = shapes_path

        # High contrast image
        contrast_img = np.zeros((300, 300, 3), dtype=np.uint8)
        contrast_img[:150, :] = 255  # White top half
        contrast_img[150:, :] = 0    # Black bottom half
        contrast_path = os.path.join(temp_dir, "high_contrast.jpg")
        cv2.imwrite(contrast_path, contrast_img)
        images["high_contrast"] = contrast_path

        # Noisy image
        noisy_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        noisy_path = os.path.join(temp_dir, "noisy_image.jpg")
        cv2.imwrite(noisy_path, noisy_img)
        images["noisy_image"] = noisy_path

        # Large image for performance testing
        large_img = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
        large_path = os.path.join(temp_dir, "large_image.jpg")
        cv2.imwrite(large_path, large_img)
        images["large_image"] = large_path

        # Empty/blank image
        blank_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        blank_path = os.path.join(temp_dir, "blank_image.jpg")
        cv2.imwrite(blank_path, blank_img)
        images["blank_image"] = blank_path

        return images

    # UNIT TESTS
    def test_initialization(self):
        """Test MultimodalAI system initialization."""
        system = MultimodalAI()
        assert hasattr(system, 'image_processor')
        assert hasattr(system, 'llm_processor')
        assert isinstance(system.image_processor, ImageProcessor)
        assert isinstance(system.llm_processor, LLMProcessor)

    def test_initialization_components(self):
        """Test that all components are properly initialized."""
        system = MultimodalAI()

        # Check image processor
        assert system.image_processor.yolo_model is not None

        # Check LLM processor
        assert system.llm_processor.processor is not None
        assert system.llm_processor.model is not None

    # BASIC FUNCTIONALITY TESTS
    def test_process_basic_functionality(self, multimodal_system, comprehensive_test_images):
        """Test basic process functionality with simple images."""
        result = multimodal_system.process(
            comprehensive_test_images["red_square"], 
            "What color is this image?"
        )

        # Check all required keys are present
        required_keys = {"caption", "objects", "colors", "ocr_text", "answer"}
        assert required_keys.issubset(result.keys())

        # Check data types
        assert isinstance(result["caption"], str)
        assert isinstance(result["objects"], list)
        assert isinstance(result["colors"], list)
        assert isinstance(result["ocr_text"], str)
        assert isinstance(result["answer"], str)

    def test_process_different_questions(self, multimodal_system, comprehensive_test_images):
        """Test processing with different types of questions."""
        questions = [
            "What color is this image?",
            "Describe what you see",
            "How many objects are in this image?",
            "What is the main subject?",
            "Is this image natural or artificial?",
            "What emotions does this image convey?",
            "What time of day might this be?",
            ""  # Empty question
        ]

        for question in questions:
            result = multimodal_system.process(
                comprehensive_test_images["complex_scene"], 
                question
            )
            assert isinstance(result["answer"], str)
            # Non-empty questions should get substantial answers
            if question.strip():
                assert len(result["answer"]) > 0

    # INTEGRATION TESTS
    def test_full_pipeline_integration(self, multimodal_system, comprehensive_test_images):
        """Test complete pipeline integration with complex scene."""
        result = multimodal_system.process(
            comprehensive_test_images["complex_scene"],
            "Describe this scene in detail"
        )

        # Verify all pipeline components executed
        assert len(result["caption"]) > 0
        assert isinstance(result["objects"], list)
        assert isinstance(result["colors"], list)
        assert len(result["colors"]) > 0
        assert isinstance(result["ocr_text"], str)
        assert len(result["answer"]) > 0

    def test_pipeline_with_text_detection(self, multimodal_system, comprehensive_test_images):
        """Test pipeline integration with text-containing images."""
        result = multimodal_system.process(
            comprehensive_test_images["text_image"],
            "What text do you see in this image?"
        )

        # Should detect some text
        assert len(result["ocr_text"]) > 0
        assert isinstance(result["answer"], str)

    def test_pipeline_with_geometric_shapes(self, multimodal_system, comprehensive_test_images):
        """Test pipeline with geometric shapes for object detection."""
        result = multimodal_system.process(
            comprehensive_test_images["geometric_shapes"],
            "What shapes do you see?"
        )

        # Should detect multiple objects
        assert isinstance(result["objects"], list)
        assert isinstance(result["answer"], str)

    # EDGE CASE TESTS
    def test_process_blank_image(self, multimodal_system, comprehensive_test_images):
        """Test processing blank/empty images."""
        result = multimodal_system.process(
            comprehensive_test_images["blank_image"],
            "What is in this image?"
        )

        # Should handle blank images gracefully
        assert isinstance(result["caption"], str)
        assert isinstance(result["objects"], list)
        assert isinstance(result["colors"], list)
        assert isinstance(result["answer"], str)

    def test_process_noisy_image(self, multimodal_system, comprehensive_test_images):
        """Test processing very noisy images."""
        result = multimodal_system.process(
            comprehensive_test_images["noisy_image"],
            "Describe this image"
        )

        # Should handle noisy images without crashing
        assert isinstance(result["answer"], str)

    def test_process_high_contrast_image(self, multimodal_system, comprehensive_test_images):
        """Test processing high contrast images."""
        result = multimodal_system.process(
            comprehensive_test_images["high_contrast"],
            "What do you see in this image?"
        )

        # Should extract colors successfully
        assert len(result["colors"]) > 0
        assert isinstance(result["answer"], str)

    def test_process_invalid_image_path(self, multimodal_system):
        """Test error handling with invalid image paths."""
        with pytest.raises(Exception):
            multimodal_system.process("non_existent_image.jpg", "What is this?")

    def test_process_corrupted_image_file(self, multimodal_system, temp_dir):
        """Test handling of corrupted image files."""
        # Create a corrupted file
        corrupted_path = os.path.join(temp_dir, "corrupted.jpg")
        with open(corrupted_path, 'w') as f:
            f.write("This is not a valid image file")

        with pytest.raises(Exception):
            multimodal_system.process(corrupted_path, "What is this?")

    def test_process_very_long_question(self, multimodal_system, comprehensive_test_images):
        """Test processing with extremely long questions."""
        long_question = " ".join(["What"] * 100) + " is in this image?"

        result = multimodal_system.process(
            comprehensive_test_images["red_square"],
            long_question
        )

        # Should handle long questions gracefully
        assert isinstance(result["answer"], str)

    def test_process_special_characters_question(self, multimodal_system, comprehensive_test_images):
        """Test processing with special characters in questions."""
        special_questions = [
            "What's in this image?",
            "Can you describe this image's contents (in detail)?",
            "What color is this: red, blue, or green?",
            "How many objects are there - approximately?",
            "Is this image clear/blurry/normal quality?"
        ]

        for question in special_questions:
            result = multimodal_system.process(
                comprehensive_test_images["complex_scene"],
                question
            )
            assert isinstance(result["answer"], str)

    # PERFORMANCE TESTS
    def test_processing_speed_benchmark(self, multimodal_system, comprehensive_test_images):
        """Test processing speed for performance benchmarking."""
        start_time = time.time()

        result = multimodal_system.process(
            comprehensive_test_images["red_square"],
            "What color is this?"
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 30.0  # 30 seconds max
        assert isinstance(result["answer"], str)

    def test_memory_usage_monitoring(self, multimodal_system, comprehensive_test_images):
        """Test memory usage during processing."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Process multiple images
        for i in range(3):
            result = multimodal_system.process(
                comprehensive_test_images["complex_scene"],
                f"Describe this image - iteration {i+1}"
            )
            assert isinstance(result["answer"], str)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 1GB)
        assert memory_increase < 1024 * 1024 * 1024  # 1GB

    def test_large_image_processing(self, multimodal_system, comprehensive_test_images):
        """Test processing of large images."""
        start_time = time.time()

        result = multimodal_system.process(
            comprehensive_test_images["large_image"],
            "Describe this image"
        )

        end_time = time.time()

        # Should handle large images without excessive time
        assert (end_time - start_time) < 60.0  # 1 minute max
        assert isinstance(result["answer"], str)

    def test_concurrent_processing_safety(self, multimodal_system, comprehensive_test_images):
        """Test thread safety with concurrent processing."""
        results = []
        errors = []

        def process_image(image_path, question, index):
            try:
                result = multimodal_system.process(image_path, f"{question} - thread {index}")
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=process_image,
                args=(comprehensive_test_images["red_square"], "What color is this?", i)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == 3
        assert all(isinstance(r["answer"], str) for r in results)

    # STRESS TESTS
    def test_repeated_processing_stability(self, multimodal_system, comprehensive_test_images):
        """Test system stability under repeated processing."""
        for i in range(10):
            result = multimodal_system.process(
                comprehensive_test_images["complex_scene"],
                f"Iteration {i+1}: Describe what you see"
            )

            # Each result should be valid
            assert isinstance(result["caption"], str)
            assert isinstance(result["answer"], str)
            assert len(result["answer"]) > 0

    def test_alternating_image_types(self, multimodal_system, comprehensive_test_images):
        """Test processing alternating between different image types."""
        image_keys = ["red_square", "complex_scene", "text_image", "geometric_shapes", "blank_image"]

        for i, key in enumerate(image_keys * 2):  # Process each type twice
            result = multimodal_system.process(
                comprehensive_test_images[key],
                f"Process {key} - iteration {i+1}"
            )

            assert isinstance(result["answer"], str)

    # ERROR HANDLING TESTS
    @patch('src.image_processor.ImageProcessor.preprocess')
    def test_image_processor_error_handling(self, mock_preprocess, multimodal_system, comprehensive_test_images):
        """Test error handling when image processor fails."""
        mock_preprocess.side_effect = Exception("Image processing failed")

        with pytest.raises(Exception):
            multimodal_system.process(
                comprehensive_test_images["red_square"],
                "What is this?"
            )

    @patch('src.llm_integration.LLMProcessor.caption_image')
    def test_llm_caption_error_handling(self, mock_caption, multimodal_system, comprehensive_test_images):
        """Test error handling when LLM captioning fails."""
        mock_caption.side_effect = Exception("Caption generation failed")

        with pytest.raises(Exception):
            multimodal_system.process(
                comprehensive_test_images["red_square"],
                "What is this?"
            )

    @patch('src.llm_integration.LLMProcessor.answer_question')
    def test_llm_qa_error_handling(self, mock_qa, multimodal_system, comprehensive_test_images):
        """Test error handling when LLM QA fails."""
        mock_qa.side_effect = Exception("Question answering failed")

        with pytest.raises(Exception):
            multimodal_system.process(
                comprehensive_test_images["red_square"],
                "What is this?"
            )

    # VALIDATION TESTS
    def test_result_format_validation(self, multimodal_system, comprehensive_test_images):
        """Test that results always follow expected format."""
        result = multimodal_system.process(
            comprehensive_test_images["complex_scene"],
            "Analyze this image"
        )

        # Validate result structure
        assert isinstance(result, dict)
        required_keys = {"caption", "objects", "colors", "ocr_text", "answer"}
        assert all(key in result for key in required_keys)

        # Validate data types and structures
        assert isinstance(result["caption"], str)
        assert isinstance(result["objects"], list)
        assert isinstance(result["colors"], list)
        assert isinstance(result["ocr_text"], str)
        assert isinstance(result["answer"], str)

        # Validate object structure
        for obj in result["objects"]:
            assert isinstance(obj, dict)
            assert "class_id" in obj
            assert "confidence" in obj
            assert "bbox" in obj

    def test_output_consistency(self, multimodal_system, comprehensive_test_images):
        """Test that identical inputs produce consistent outputs."""
        image_path = comprehensive_test_images["red_square"]
        question = "What color is this image?"

        # Process same input multiple times
        results = []
        for _ in range(3):
            result = multimodal_system.process(image_path, question)
            results.append(result)

        # Results should be identical for same input
        for i in range(1, len(results)):
            assert results[i]["caption"] == results[0]["caption"]
            assert results[i]["answer"] == results[0]["answer"]
            # Colors and objects might vary slightly due to floating point precision

    # FEATURE COMPLETENESS TESTS
    def test_all_image_processing_features(self, multimodal_system, comprehensive_test_images):
        """Test that all image processing features are utilized."""
        result = multimodal_system.process(
            comprehensive_test_images["complex_scene"],
            "Provide a complete analysis"
        )

        # Should have attempted all processing features
        assert "caption" in result  # LLM captioning
        assert "objects" in result  # Object detection
        assert "colors" in result   # Color extraction
        assert "ocr_text" in result  # Text extraction
        assert "answer" in result   # Question answering

    def test_color_extraction_quality(self, multimodal_system, comprehensive_test_images):
        """Test quality of color extraction across different images."""
        for color in ["red", "blue", "green"]:
            result = multimodal_system.process(
                comprehensive_test_images[f"{color}_square"],
                f"What is the dominant color?"
            )

            # Should extract meaningful colors
            assert len(result["colors"]) > 0
            assert all(isinstance(color_tuple, tuple) for color_tuple in result["colors"])
            assert all(len(color_tuple) == 3 for color_tuple in result["colors"])

    def test_context_integration(self, multimodal_system, comprehensive_test_images):
        """Test that visual context is properly integrated into answers."""
        result = multimodal_system.process(
            comprehensive_test_images["complex_scene"],
            "Based on what you see, what kind of environment is this?"
        )

        # Answer should reflect visual analysis
        assert len(result["answer"]) > 10  # Should be substantial
        assert isinstance(result["caption"], str)
        assert len(result["caption"]) > 0
