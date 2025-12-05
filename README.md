# Multimodal AI System

A comprehensive multimodal AI system that combines computer vision and natural language processing using OpenCV, YOLOv8, and BLIP models. Upload an image, ask questions, and get intelligent answers powered by state-of-the-art deep learning.

## Features

- **Object Detection**: Real-time object detection using YOLOv8 with bounding boxes and confidence scores
- **Image Captioning**: Automatic image description using Salesforce BLIP vision-language model
- **Visual Question Answering**: Ask natural language questions about image content
- **Color Analysis**: K-means clustering to extract dominant colors
- **Text Recognition (OCR)**: Tesseract-based optical character recognition
- **Web Interface**: Interactive Streamlit application with drag-and-drop upload
- **Comprehensive Testing**: Full pytest test suite with unit and integration tests

## Project Structure

```
multimodal_ai_project/
├── src/
│   ├── image_processor.py      # OpenCV + YOLOv8 + OCR processing
│   ├── llm_integration.py      # BLIP vision-language model wrapper
│   ├── multimodal_system.py    # Main orchestrator combining all components
│   ├── web_app.py              # Streamlit web interface
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
| opencv-python | 4.10.0 | Image processing |
| torch | 2.7.1 | Deep learning framework |
| transformers | 4.53.3 | BLIP model loading |
| ultralytics | 8.3.26 | YOLOv8 object detection |
| streamlit | 1.47.0 | Web interface |
| pytesseract | 0.3.13 | OCR text extraction |
| Pillow | 10.4.0 | Image I/O |

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_image_processor.py -v
```

## Core Components

### ImageProcessor

Handles all computer vision tasks using OpenCV:

- Image loading, resizing, and normalization
- Object detection with YOLOv8 (supports nano/small/medium/large models)
- Color extraction using K-means clustering
- Text extraction with Tesseract OCR preprocessing

### LLMProcessor

Wraps the Salesforce BLIP vision-language model:

- Conditional and unconditional image captioning
- Visual question answering based on image context
- Automatic device detection (CPU/CUDA/MPS)
- Batch processing support

### MultimodalAI

Main orchestrator that combines all components:

1. Preprocess image (resize, normalize)
2. Extract visual features (objects, colors, text)
3. Generate image caption
4. Answer questions using combined context
5. Generate analysis summary

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

## Web Interface Features

- **Image Upload**: Drag-and-drop or file browser
- **Question Input**: Natural language questions about images
- **Tabbed Results**: Organized display of AI response, objects, colors, and text
- **Configuration Sidebar**: Adjust confidence thresholds, color count, model parameters
- **Detection Visualization**: Bounding boxes overlaid on images
- **JSON Export**: Download complete analysis results

## Example Output

```json
{
  "caption": {
    "caption": "a dog sitting on a couch in a living room",
    "confidence": 0.85
  },
  "features": {
    "objects": [
      {"class_name": "dog", "confidence": 0.92, "bbox": [120, 80, 340, 290]},
      {"class_name": "couch", "confidence": 0.87, "bbox": [50, 150, 450, 380]}
    ],
    "colors": [
      {"rgb": [139, 90, 43], "hex": "#8b5a2b", "percentage": 28.5},
      {"rgb": [210, 180, 140], "hex": "#d2b48c", "percentage": 22.1}
    ],
    "ocr_text": {"text": "", "word_count": 0}
  },
  "answer": {
    "answer": "I can see a dog sitting on a couch",
    "confidence": 0.78
  }
}
```

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