# Multimodal AI System

A comprehensive multimodal AI system that combines computer vision and natural language processing using OpenCV, YOLOv8, and BLIP models.

## 🌟 Features

- **Image Processing**: OpenCV-based preprocessing, object detection with YOLOv8, color analysis, and OCR
- **Vision-Language Model**: BLIP-based image captioning and visual question answering
- **Web Interface**: User-friendly Streamlit application for interactive analysis
- **Production Ready**: Complete test suite, error handling, and pip-installable package

## 📁 Project Structure

```
multimodal_ai_project/
├── src/
│   ├── image_processor.py      # OpenCV + YOLOv8 + OCR processing
│   ├── llm_integration.py      # BLIP vision-language model wrapper
│   ├── multimodal_system.py    # Main system integration logic
│   ├── web_app.py              # Streamlit web interface
│   ├── utils.py                # Shared helper functions
│   └── __init__.py             # Package initialization
├── data/                       # Sample images directory + README
├── tests/                      # Complete pytest test suite (5 files)
├── requirements.txt            # Pinned versions for reproducibility
├── setup.py                    # Pip-installable package configuration
├── run_app.py                  # Quick launch script for web interface
├── example_usage.py            # Programmatic usage examples
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd multimodal_ai_project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch Web Interface

```bash
python run_app.py
```

Or alternatively:
```bash
streamlit run src/web_app.py
```

### 3. Programmatic Usage

```python
from src.multimodal_system import MultimodalAI

# Initialize the system
system = MultimodalAI(
    yolo_model="yolov8n.pt",
    blip_model="Salesforce/blip-image-captioning-base",
    device="auto"
)

# Process an image with a question
results = system.process(
    "data/your_image.jpg",
    "What objects do you see in this image?"
)

print(results['answer'])
```

## 🔧 System Requirements

- **Python**: 3.8 or higher
- **GPU**: Optional but recommended for faster processing
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: ~2GB for models and dependencies

## 🧪 Testing

Run the complete test suite:

```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📊 Core Components

### ImageProcessor
- **YOLOv8 Object Detection**: Real-time object detection with confidence scoring
- **Color Analysis**: K-means clustering and histogram analysis
- **OCR Text Extraction**: Tesseract-based text recognition
- **Image Preprocessing**: OpenCV resizing, normalization, and format conversion

### LLMProcessor
- **Image Captioning**: BLIP-based conditional and unconditional captioning
- **Visual Q&A**: Context-aware question answering about image content
- **Batch Processing**: Efficient processing of multiple images
- **Device Optimization**: Automatic GPU/CPU detection and optimization

### MultimodalAI
- **Unified Pipeline**: Combines all components into a single interface
- **Feature Integration**: Merges visual features with language understanding
- **Result Export**: JSON export functionality for analysis results
- **Configuration Management**: Flexible model and device configuration

## 🌐 Web Interface Features

- **Drag & Drop Upload**: Easy image uploading
- **Interactive Q&A**: Ask questions about uploaded images
- **Tabbed Results**: Organized display of AI responses, detected objects, colors, and text
- **Real-time Configuration**: Adjust confidence thresholds and other parameters
- **Detection Visualization**: Bounding box overlays for detected objects
- **Export Functionality**: Download results as JSON

## 🛠️ Configuration Options

### YOLO Models
- `yolov8n.pt`: Nano (fastest, least accurate)
- `yolov8s.pt`: Small (balanced)
- `yolov8m.pt`: Medium (more accurate)
- `yolov8l.pt`: Large (most accurate, slowest)

### BLIP Models
- `Salesforce/blip-image-captioning-base`: Base model (default)
- `Salesforce/blip-image-captioning-large`: Large model (more accurate)
- `Salesforce/blip-vqa-base`: Specialized for visual Q&A

### Device Options
- `auto`: Automatically detect best available device
- `cpu`: Force CPU usage
- `cuda`: Force GPU usage (if available)

## 📈 Performance Tips

1. **GPU Usage**: Enable CUDA for 3-5x faster processing
2. **Model Selection**: Use nano models for real-time applications
3. **Image Size**: Optimal size is 640x640 pixels for YOLO
4. **Batch Processing**: Process multiple images together for efficiency

## 🔍 Example Outputs

### Object Detection
```json
{
  "objects": [
    {"class": "person", "confidence": 0.89, "bbox": [100, 150, 200, 300]},
    {"class": "car", "confidence": 0.76, "bbox": [300, 200, 500, 400]}
  ]
}
```

### Color Analysis
```json
{
  "dominant_colors": [
    {"color": [120, 80, 160], "percentage": 35.2},
    {"color": [200, 180, 140], "percentage": 28.7}
  ]
}
```

### OCR Results
```json
{
  "text_regions": [
    {"text": "STOP", "confidence": 0.95, "bbox": [50, 60, 150, 100]},
    {"text": "Main Street", "confidence": 0.87, "bbox": [200, 300, 400, 330]}
  ]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Model Download Errors**:
```bash
# Download models manually
pip install huggingface_hub
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**CUDA Out of Memory**:
```python
# Use CPU or smaller model
system = MultimodalAI(yolo_model="yolov8n.pt", device="cpu")
```

**Import Errors**:
```bash
# Install in development mode
pip install -e .
```

## 🎯 Roadmap

- [ ] Support for video processing
- [ ] Multi-language OCR support
- [ ] Custom model training pipeline
- [ ] REST API interface
- [ ] Docker containerization
- [ ] Cloud deployment templates

---

Built with ❤️ using OpenCV, YOLOv8, BLIP, and Streamlit