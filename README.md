# Multimodal AI System

A comprehensive multimodal AI system that combines computer vision and natural language processing using OpenCV, YOLOv8, and BLIP models. Upload an image, ask questions, and get intelligent answers powered by state-of-the-art deep learning.

## Features

### Core Analysis
- **Object Detection**: Real-time object detection using YOLOv8 with bounding boxes and confidence scores
- **Image Captioning**: Automatic image description using Salesforce BLIP vision-language model
- **Visual Question Answering**: Ask natural language questions about image content
- **Color Analysis**: K-means clustering to extract dominant colors with hex codes
- **Text Recognition (OCR)**: Tesseract-based optical character recognition

### New Features
- **Face Detection**: Automatic face detection with location and size information
- **Image Quality Scoring**: Brightness, contrast, sharpness, and blur detection metrics
- **Dark/Light Theme**: Toggle between dark and light modes
- **URL Image Loading**: Load images directly from web URLs
- **Image History**: Quick access to last 5 analyzed images
- **Suggested Questions**: AI-powered question suggestions based on image content
- **Modern UI**: Glass-morphism design with smooth gradients and animations

## Project Structure

```
multimodal_ai_project/
├── src/
│   ├── image_processor.py      # OpenCV + YOLOv8 + OCR processing
│   ├── llm_integration.py      # BLIP vision-language model wrapper
│   ├── multimodal_system.py    # Main orchestrator combining all components
│   ├── web_app.py              # Enhanced Streamlit web interface
│   ├── utils.py                # Shared utility functions
│   └── __init__.py             # Package exports
├── tests/
│   ├── test_image_processor.py
│   ├── test_llm_integration.py
│   └── test_multimodal_system.py
├── data/                       # Sample images directory
├── requirements.txt            # Pinned dependency versions
├── setup.py                    # Package configuration
├── .gitignore                  # Git ignore patterns
└── README.md
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/multimodal_ai_project.git
cd multimodal_ai_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Launch Web Interface

```bash
streamlit run src/web_app.py
```

The application will be available at `http://localhost:8501`

### Programmatic Usage

```python
from src.multimodal_system import MultimodalAI

# Initialize the system
system = MultimodalAI(
    yolo_model="yolov8n.pt",
    blip_model="Salesforce/blip-image-captioning-base",
    device="auto"  # Automatically uses GPU if available
)

# Process an image with a question
results = system.process(
    "path/to/image.jpg",
    "What objects do you see in this image?"
)

# Access results
print("Caption:", results['caption']['caption'])
print("Answer:", results['answer']['answer'])
print("Detected Objects:", results['features']['objects'])
print("Dominant Colors:", results['features']['colors'])
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB | 5 GB |
| GPU | Optional | NVIDIA CUDA-enabled |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.10.0 | Image processing & face detection |
| torch | 2.7.1 | Deep learning framework |
| transformers | 4.53.3 | BLIP model loading |
| ultralytics | 8.3.26 | YOLOv8 object detection |
| streamlit | 1.47.0 | Web interface |
| pytesseract | 0.3.13 | OCR text extraction |
| Pillow | 10.4.0 | Image I/O |
| requests | 2.32.0 | URL image loading |

## Web Interface Features

### Theme Support
- Dark mode with purple gradients
- Light mode with clean, professional styling
- Theme preference persists during session

### Input Methods
- File upload (drag-and-drop or browse)
- URL loading (paste any image URL)

### Analysis Options (Configurable)
- Object detection with confidence threshold
- Face detection
- Color analysis (3-10 colors)
- Text recognition (OCR)
- Image quality scoring

### Results Display
- Tabbed interface: Objects, Faces, Colors, Text, Quality
- Annotated image with bounding boxes
- Suggested follow-up questions
- JSON export for all results

## Image Quality Metrics

The quality analyzer provides:

| Metric | Description |
|--------|-------------|
| Brightness | Average pixel intensity (0-255) |
| Contrast | Standard deviation of pixels |
| Sharpness | Laplacian variance (higher = sharper) |
| Blur Detection | Boolean indicating if image is blurry |
| Overall Score | Weighted combination (0-100) |

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_image_processor.py -v
```

## Configuration Options

### YOLO Models

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| yolov8n.pt | Fastest | Lower | Real-time applications |
| yolov8s.pt | Fast | Balanced | General purpose |
| yolov8m.pt | Medium | Higher | Detailed analysis |
| yolov8l.pt | Slower | Highest | Maximum accuracy |

### BLIP Models

- `Salesforce/blip-image-captioning-base` - Default, balanced
- `Salesforce/blip-image-captioning-large` - More accurate, slower
- `Salesforce/blip-vqa-base` - Optimized for Q&A tasks

## Troubleshooting

### Model Download Errors

```bash
pip install huggingface_hub
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### CUDA Out of Memory

```python
# Use CPU or smaller model
system = MultimodalAI(yolo_model="yolov8n.pt", device="cpu")
```

### Import Errors

```bash
# Install package in development mode
pip install -e .
```

### Tesseract Not Found

```bash
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

## License

This project is licensed under the MIT License.

---

Built with OpenCV, YOLOv8, BLIP, and Streamlit