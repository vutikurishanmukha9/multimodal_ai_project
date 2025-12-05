"""
Comprehensive unit tests for ImageProcessor class.
Tests image preprocessing, object detection, color extraction, and OCR functionality.
"""

import os
import pytest
import numpy as np
import cv2
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from src.image_processor import ImageProcessor


class TestImageProcessor:
    """Test suite for ImageProcessor class."""

    @pytest.fixture(scope="class")
    def processor(self):
        """Fixture that returns an initialized ImageProcessor."""
        return ImageProcessor()

    @pytest.fixture(scope="function")
    def temp_dir(self):
        """Create temporary directory for test images."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture(scope="function")
    def sample_images(self, temp_dir):
        """Create various sample images for testing."""
        images = {}

        # Create a simple color image
        color_img = np.zeros((300, 400, 3), dtype=np.uint8)
        color_img[:, :200] = [255, 0, 0]  # Red half
        color_img[:, 200:] = [0, 255, 0]  # Green half
        color_path = os.path.join(temp_dir, "color_test.jpg")
        cv2.imwrite(color_path, color_img)
        images['color'] = color_path

        # Create an image with text
        text_img = 255 * np.ones((200, 600, 3), dtype=np.uint8)
        cv2.putText(text_img, "Hello World Test", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        text_path = os.path.join(temp_dir, "text_test.jpg")
        cv2.imwrite(text_path, text_img)
        images['text'] = text_path

        # Create a blank white image
        blank_img = 255 * np.ones((100, 100, 3), dtype=np.uint8)
        blank_path = os.path.join(temp_dir, "blank_test.jpg")
        cv2.imwrite(blank_path, blank_img)
        images['blank'] = blank_path

        # Create a complex pattern image for object detection
        pattern_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        # Add some geometric shapes
        cv2.rectangle(pattern_img, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(pattern_img, (400, 400), 50, (0, 0, 255), -1)
        pattern_path = os.path.join(temp_dir, "pattern_test.jpg")
        cv2.imwrite(pattern_path, pattern_img)
        images['pattern'] = pattern_path

        return images

    def test_initialization(self):
        """Test ImageProcessor initialization."""
        processor = ImageProcessor()
        assert processor.yolo_model is not None
        assert hasattr(processor, 'yolo_model')

    def test_initialization_custom_model(self):
        """Test ImageProcessor initialization with custom model."""
        processor = ImageProcessor(yolo_model='yolov8s.pt')
        assert processor.yolo_model is not None

    @pytest.mark.parametrize("size", [(640, 640), (320, 320), (800, 600), (1024, 768)])
    def test_preprocess_dimensions(self, processor, sample_images, size):
        """Test image preprocessing with various dimensions."""
        img_resized, img_norm = processor.preprocess(sample_images['color'], size=size)

        assert img_resized.shape[:2] == size  # height, width
        assert img_norm.max() <= 1.0
        assert img_norm.min() >= 0.0
        assert img_norm.dtype == np.float64

    def test_preprocess_invalid_path(self, processor):
        """Test preprocessing with invalid image path."""
        with pytest.raises(Exception):
            processor.preprocess("non_existent_image.jpg")

    def test_preprocess_corrupted_image(self, processor, temp_dir):
        """Test preprocessing with corrupted image file."""
        # Create a corrupted file
        corrupted_path = os.path.join(temp_dir, "corrupted.jpg")
        with open(corrupted_path, 'w') as f:
            f.write("This is not an image")

        with pytest.raises(Exception):
            processor.preprocess(corrupted_path)

    def test_detect_objects_basic(self, processor, sample_images):
        """Test basic object detection functionality."""
        img, _ = processor.preprocess(sample_images['pattern'])
        objects = processor.detect_objects(img)

        assert isinstance(objects, list)
        for obj in objects:
            assert 'class_id' in obj
            assert 'confidence' in obj
            assert 'bbox' in obj
            assert isinstance(obj['class_id'], int)
            assert isinstance(obj['confidence'], float)
            assert isinstance(obj['bbox'], list)
            assert len(obj['bbox']) == 4
            assert 0.0 <= obj['confidence'] <= 1.0

    def test_detect_objects_empty_image(self, processor, sample_images):
        """Test object detection on blank image."""
        img, _ = processor.preprocess(sample_images['blank'])
        objects = processor.detect_objects(img)

        assert isinstance(objects, list)
        # Blank image should have few or no detections

    @patch('cv2.kmeans')
    def test_extract_colors_basic(self, mock_kmeans, processor, sample_images):
        """Test color extraction functionality."""
        # Mock cv2.kmeans to return predictable results
        mock_centers = np.array([[100, 150, 200], [50, 75, 100]], dtype=np.uint8)
        mock_kmeans.return_value = (None, None, mock_centers)

        img, _ = processor.preprocess(sample_images['color'])
        colors = processor.extract_colors(img, k=2)

        assert isinstance(colors, list)
        assert len(colors) == 2
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(isinstance(c, int) for c in color)

    @pytest.mark.parametrize("k", [3, 5, 8, 10])
    def test_extract_colors_different_k(self, processor, sample_images, k):
        """Test color extraction with different k values."""
        img, _ = processor.preprocess(sample_images['color'])
        colors = processor.extract_colors(img, k=k)

        assert isinstance(colors, list)
        assert len(colors) == k

    def test_extract_colors_invalid_k(self, processor, sample_images):
        """Test color extraction with invalid k values."""
        img, _ = processor.preprocess(sample_images['color'])

        # k must be positive
        with pytest.raises(Exception):
            processor.extract_colors(img, k=0)

        with pytest.raises(Exception):
            processor.extract_colors(img, k=-1)

    @patch('pytesseract.image_to_string')
    def test_extract_text_basic(self, mock_ocr, processor, sample_images):
        """Test OCR text extraction functionality."""
        mock_ocr.return_value = "Hello World Test"

        img, _ = processor.preprocess(sample_images['text'])
        text = processor.extract_text(img)

        assert isinstance(text, str)
        assert text == "Hello World Test"
        mock_ocr.assert_called_once()

    @patch('pytesseract.image_to_string')
    def test_extract_text_empty_result(self, mock_ocr, processor, sample_images):
        """Test OCR with no text found."""
        mock_ocr.return_value = "   \n\n   "  # Whitespace and newlines

        img, _ = processor.preprocess(sample_images['blank'])
        text = processor.extract_text(img)

        assert text == ""  # Should be stripped to empty string

    @patch('pytesseract.image_to_string')
    def test_extract_text_ocr_error(self, mock_ocr, processor, sample_images):
        """Test OCR error handling."""
        mock_ocr.side_effect = Exception("OCR failed")

        img, _ = processor.preprocess(sample_images['text'])

        with pytest.raises(Exception):
            processor.extract_text(img)

    def test_full_pipeline_integration(self, processor, sample_images):
        """Test complete image processing pipeline."""
        img_path = sample_images['pattern']

        # Preprocess
        img, norm_img = processor.preprocess(img_path)
        assert img.shape[:2] == (640, 640)

        # Detect objects
        objects = processor.detect_objects(img)
        assert isinstance(objects, list)

        # Extract colors
        colors = processor.extract_colors(img)
        assert isinstance(colors, list)
        assert len(colors) == 5  # default k=5

        # Extract text (may be empty for pattern image)
        text = processor.extract_text(img)
        assert isinstance(text, str)

    def test_memory_efficiency(self, processor, temp_dir):
        """Test processing of large images for memory efficiency."""
        # Create a large image
        large_img = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        large_path = os.path.join(temp_dir, "large_test.jpg")
        cv2.imwrite(large_path, large_img)

        # Should handle large images without crashing
        img, norm_img = processor.preprocess(large_path, size=(640, 640))
        assert img.shape[:2] == (640, 640)

        # Clean up explicitly
        del large_img, img, norm_img

    def test_color_space_conversion(self, processor, sample_images):
        """Test that color extraction properly converts color spaces."""
        img, _ = processor.preprocess(sample_images['color'])
        colors = processor.extract_colors(img, k=3)

        # Colors should be in LAB space as tuples
        for color in colors:
            assert len(color) == 3
            # LAB values should be in reasonable ranges
            assert 0 <= color[0] <= 255  # L channel
            assert 0 <= color[1] <= 255  # A channel  
            assert 0 <= color[2] <= 255  # B channel

    @pytest.mark.parametrize("image_format", [".jpg", ".png", ".jpeg"])
    def test_different_image_formats(self, processor, temp_dir, image_format):
        """Test processing different image file formats."""
        # Create test image in different formats
        test_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        img_path = os.path.join(temp_dir, f"test{image_format}")
        cv2.imwrite(img_path, test_img)

        # Should handle different formats
        img, norm_img = processor.preprocess(img_path)
        assert img.shape[:2] == (640, 640)

    def test_object_detection_confidence_filtering(self, processor, sample_images):
        """Test that object detection returns valid confidence scores."""
        img, _ = processor.preprocess(sample_images['pattern'])
        objects = processor.detect_objects(img)

        for obj in objects:
            confidence = obj['confidence']
            assert 0.0 <= confidence <= 1.0
            # Most detections should have reasonable confidence
            assert confidence > 0.1  # Very low confidence detections are suspicious

    def test_bbox_coordinates_validity(self, processor, sample_images):
        """Test that bounding box coordinates are valid."""
        img, _ = processor.preprocess(sample_images['pattern'])
        objects = processor.detect_objects(img)

        h, w = img.shape[:2]
        for obj in objects:
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox

            # Coordinates should be within image bounds
            assert 0 <= x1 < w
            assert 0 <= y1 < h
            assert 0 <= x2 <= w
            assert 0 <= y2 <= h

            # x2 > x1 and y2 > y1 (valid rectangle)
            assert x2 > x1
            assert y2 > y1

    def test_processor_reusability(self, processor, sample_images):
        """Test that processor can be reused for multiple images."""
        # Process multiple images with same processor instance
        for img_key in ['color', 'text', 'blank']:
            img, _ = processor.preprocess(sample_images[img_key])
            objects = processor.detect_objects(img)
            colors = processor.extract_colors(img)
            text = processor.extract_text(img)

            # All should return valid results
            assert isinstance(objects, list)
            assert isinstance(colors, list)
            assert isinstance(text, str)
