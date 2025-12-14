"""
Enhanced Streamlit web application for the Multimodal AI System.

Features:
- Dark/Light theme toggle
- Modern glass-morphism UI
- Image history gallery
- URL image loading
- Face detection
- Image quality scoring
- Progress indicators
- Suggested questions
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
import logging
from typing import Dict, Any, Optional, List
import tempfile
import os
import sys
import requests
from datetime import datetime

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Add parent directory to path for Streamlit execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
try:
    from src.multimodal_system import MultimodalAI
    from src.utils import convert_bgr_to_rgb
except ImportError:
    from multimodal_system import MultimodalAI
    from utils import convert_bgr_to_rgb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multimodal AI System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_theme_css(dark_mode: bool) -> str:
    """Generate CSS based on theme selection."""
    if dark_mode:
        return """
        <style>
            :root {
                --bg-primary: #1e2432;
                --bg-secondary: #2a3142;
                --bg-card: rgba(45, 55, 75, 0.9);
                --text-primary: #f5f5f5;
                --text-secondary: #c5cdd8;
                --accent: #6366f1;
                --accent-hover: #818cf8;
                --border: rgba(255, 255, 255, 0.15);
                --success: #10b981;
                --warning: #f59e0b;
                --error: #ef4444;
            }
            
            .stApp {
                background: linear-gradient(135deg, #1e2432 0%, #2a3142 50%, #1e2432 100%);
            }
            
            .main-header {
                font-size: 2.8em;
                background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 0.5em;
                font-weight: 700;
            }
            
            .glass-card {
                background: var(--bg-card);
                backdrop-filter: blur(10px);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1.5em;
                margin: 1em 0;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            
            .feature-badge {
                display: inline-block;
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                color: white;
                padding: 0.3em 0.8em;
                border-radius: 20px;
                font-size: 0.85em;
                margin: 0.2em;
            }
            
            .stat-card {
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 12px;
                padding: 1em;
                text-align: center;
            }
            
            .stat-number {
                font-size: 2em;
                font-weight: 700;
                color: #8b5cf6;
            }
            
            .stat-label {
                color: var(--text-secondary);
                font-size: 0.9em;
            }
            
            .success-box {
                background: rgba(16, 185, 129, 0.15);
                border: 1px solid rgba(16, 185, 129, 0.3);
                color: #10b981;
                padding: 1em;
                border-radius: 12px;
                margin: 1em 0;
            }
            
            .warning-box {
                background: rgba(245, 158, 11, 0.15);
                border: 1px solid rgba(245, 158, 11, 0.3);
                color: #f59e0b;
                padding: 1em;
                border-radius: 12px;
                margin: 1em 0;
            }
            
            .error-box {
                background: rgba(239, 68, 68, 0.15);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #ef4444;
                padding: 1em;
                border-radius: 12px;
                margin: 1em 0;
            }
            
            .info-box {
                background: rgba(99, 102, 241, 0.15);
                border: 1px solid rgba(99, 102, 241, 0.3);
                color: var(--text-primary);
                padding: 1.5em;
                border-radius: 12px;
                margin: 1em 0;
            }
            
            .history-item {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 0.5em;
                margin: 0.3em 0;
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .history-item:hover {
                border-color: var(--accent);
                transform: translateX(5px);
            }
            
            .question-chip {
                display: inline-block;
                background: rgba(99, 102, 241, 0.2);
                border: 1px solid rgba(99, 102, 241, 0.4);
                color: var(--text-primary);
                padding: 0.4em 0.8em;
                border-radius: 20px;
                font-size: 0.85em;
                margin: 0.2em;
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .question-chip:hover {
                background: rgba(99, 102, 241, 0.4);
            }
            
            .quality-score {
                font-size: 3em;
                font-weight: 700;
            }
            
            .quality-good { color: #10b981; }
            .quality-medium { color: #f59e0b; }
            .quality-poor { color: #ef4444; }
            
            /* Progress bar styling */
            .stProgress > div > div {
                background: linear-gradient(90deg, #6366f1, #8b5cf6);
            }
        </style>
        """
    else:
        return """
        <style>
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f8fafc;
                --bg-card: rgba(255, 255, 255, 0.9);
                --text-primary: #1e293b;
                --text-secondary: #64748b;
                --accent: #6366f1;
                --accent-hover: #4f46e5;
                --border: rgba(0, 0, 0, 0.1);
                --success: #10b981;
                --warning: #f59e0b;
                --error: #ef4444;
            }
            
            .stApp {
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f8fafc 100%);
            }
            
            .main-header {
                font-size: 2.8em;
                background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 0.5em;
                font-weight: 700;
            }
            
            .glass-card {
                background: var(--bg-card);
                backdrop-filter: blur(10px);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1.5em;
                margin: 1em 0;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            }
            
            .feature-badge {
                display: inline-block;
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                color: white;
                padding: 0.3em 0.8em;
                border-radius: 20px;
                font-size: 0.85em;
                margin: 0.2em;
            }
            
            .stat-card {
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
                border: 1px solid rgba(99, 102, 241, 0.2);
                border-radius: 12px;
                padding: 1em;
                text-align: center;
            }
            
            .stat-number {
                font-size: 2em;
                font-weight: 700;
                color: #6366f1;
            }
            
            .stat-label {
                color: var(--text-secondary);
                font-size: 0.9em;
            }
            
            .success-box {
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.3);
                color: #059669;
                padding: 1em;
                border-radius: 12px;
                margin: 1em 0;
            }
            
            .warning-box {
                background: rgba(245, 158, 11, 0.1);
                border: 1px solid rgba(245, 158, 11, 0.3);
                color: #d97706;
                padding: 1em;
                border-radius: 12px;
                margin: 1em 0;
            }
            
            .error-box {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #dc2626;
                padding: 1em;
                border-radius: 12px;
                margin: 1em 0;
            }
            
            .info-box {
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid rgba(99, 102, 241, 0.2);
                color: var(--text-primary);
                padding: 1.5em;
                border-radius: 12px;
                margin: 1em 0;
            }
            
            .history-item {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 0.5em;
                margin: 0.3em 0;
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .history-item:hover {
                border-color: var(--accent);
                transform: translateX(5px);
            }
            
            .question-chip {
                display: inline-block;
                background: rgba(99, 102, 241, 0.15);
                border: 1px solid rgba(99, 102, 241, 0.3);
                color: var(--text-primary);
                padding: 0.4em 0.8em;
                border-radius: 20px;
                font-size: 0.85em;
                margin: 0.2em;
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .question-chip:hover {
                background: rgba(99, 102, 241, 0.3);
            }
            
            .quality-score {
                font-size: 3em;
                font-weight: 700;
            }
            
            .quality-good { color: #10b981; }
            .quality-medium { color: #f59e0b; }
            .quality-poor { color: #ef4444; }
            
            /* Progress bar styling */
            .stProgress > div > div {
                background: linear-gradient(90deg, #6366f1, #8b5cf6);
            }
        </style>
        """


def init_session_state():
    """Initialize session state variables."""
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False  # Default to light mode
    if 'image_history' not in st.session_state:
        st.session_state.image_history = []
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""


@st.cache_resource
def load_multimodal_system():
    """Load the multimodal AI system with caching."""
    try:
        system = MultimodalAI(
            yolo_model="yolov8n.pt",
            blip_model="Salesforce/blip-image-captioning-base",
            device="auto"
        )
        return system
    except Exception as e:
        logger.error(f"Failed to load multimodal system: {str(e)}")
        return None


def load_face_cascade():
    """Load OpenCV face detection cascade."""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        return face_cascade
    except Exception as e:
        logger.error(f"Failed to load face cascade: {str(e)}")
        return None


def detect_faces(image: np.ndarray) -> List[Dict]:
    """Detect faces in an image using OpenCV."""
    face_cascade = load_face_cascade()
    if face_cascade is None:
        return []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    results = []
    for (x, y, w, h) in faces:
        results.append({
            'bbox': (int(x), int(y), int(x + w), int(y + h)),
            'confidence': 0.85,  # Haar cascade doesn't provide confidence
            'width': int(w),
            'height': int(h)
        })
    
    return results


def calculate_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """Calculate image quality metrics."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Brightness (mean pixel value)
    brightness = np.mean(gray)
    
    # Contrast (standard deviation)
    contrast = np.std(gray)
    
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Blur detection (lower = more blurry)
    is_blurry = sharpness < 100
    
    # Overall score (0-100)
    brightness_score = min(100, max(0, 100 - abs(brightness - 127) * 0.8))
    contrast_score = min(100, contrast * 1.5)
    sharpness_score = min(100, sharpness / 10)
    
    overall_score = (brightness_score * 0.3 + contrast_score * 0.3 + sharpness_score * 0.4)
    
    return {
        'brightness': round(brightness, 2),
        'contrast': round(contrast, 2),
        'sharpness': round(sharpness, 2),
        'is_blurry': is_blurry,
        'brightness_score': round(brightness_score, 1),
        'contrast_score': round(contrast_score, 1),
        'sharpness_score': round(sharpness_score, 1),
        'overall_score': round(overall_score, 1)
    }


def load_image_from_url(url: str) -> Optional[np.ndarray]:
    """Load image from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Failed to load image from URL: {str(e)}")
        return None


def add_to_history(image_path: str, thumbnail: Image.Image, results: Dict):
    """Add analyzed image to history."""
    history_item = {
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'image_path': image_path,
        'thumbnail': thumbnail,
        'results': results
    }
    
    st.session_state.image_history.insert(0, history_item)
    
    # Keep only last 5 items
    if len(st.session_state.image_history) > 5:
        st.session_state.image_history = st.session_state.image_history[:5]


def get_suggested_questions(results: Dict) -> List[str]:
    """Generate suggested questions based on analysis results."""
    questions = []
    
    # Based on detected objects
    if results.get('features', {}).get('objects'):
        objects = results['features']['objects']
        if objects:
            class_names = list(set([obj['class_name'] for obj in objects]))
            if len(class_names) == 1:
                questions.append(f"What is the {class_names[0]} doing?")
            else:
                questions.append(f"How do the {class_names[0]} and {class_names[1]} relate?")
            questions.append("What is the main subject of this image?")
    
    # Based on colors
    if results.get('features', {}).get('colors'):
        questions.append("Why are these colors dominant in this image?")
    
    # Based on text
    if results.get('features', {}).get('ocr_text', {}).get('text'):
        questions.append("What does the text in the image say?")
    
    # Generic questions
    questions.extend([
        "Describe this image in detail",
        "What is happening in this scene?",
        "What emotions does this image convey?",
        "Where was this photo likely taken?"
    ])
    
    return questions[:6]  # Return max 6 suggestions


def display_image_with_detections(image: np.ndarray, detections: List, faces: List = None) -> np.ndarray:
    """Draw detections and faces on image."""
    result_image = image.copy()
    
    # Draw object detections
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
    ]
    
    if detections:
        for i, detection in enumerate(detections):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = detection['bbox']
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(result_image, (x1, y1 - text_h - baseline - 5), 
                         (x1 + text_w, y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - baseline - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw face detections
    if faces:
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(result_image, "Face", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return result_image


def display_quality_metrics(quality: Dict):
    """Display image quality metrics."""
    score = quality['overall_score']
    
    if score >= 70:
        score_class = "quality-good"
        label = "Good"
    elif score >= 40:
        score_class = "quality-medium"
        label = "Fair"
    else:
        score_class = "quality-poor"
        label = "Poor"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="quality-score {score_class}">{score:.0f}</div>
            <div class="stat-label">Overall ({label})</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{quality['brightness']:.0f}</div>
            <div class="stat-label">Brightness</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{quality['contrast']:.0f}</div>
            <div class="stat-label">Contrast</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        blur_text = "Yes" if quality['is_blurry'] else "No"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{blur_text}</div>
            <div class="stat-label">Blurry</div>
        </div>
        """, unsafe_allow_html=True)


def display_color_palette(colors: List):
    """Display extracted colors as a palette."""
    if not colors:
        return
    
    st.markdown("### Dominant Colors")
    
    cols = st.columns(min(len(colors), 5))
    
    for i, color in enumerate(colors[:5]):
        with cols[i]:
            hex_color = color['hex']
            percentage = color['percentage']
            
            st.markdown(f"""
                <div style="
                    background-color: {hex_color};
                    height: 80px;
                    border-radius: 10px;
                    border: 2px solid rgba(255,255,255,0.2);
                    margin-bottom: 5px;
                "></div>
                <p style="text-align: center; margin: 0;">
                    <strong>{hex_color}</strong><br>
                    {percentage:.1f}%
                </p>
            """, unsafe_allow_html=True)


def display_object_summary(objects: List):
    """Display object detection summary."""
    if not objects:
        st.info("No objects detected in the image.")
        return
    
    st.markdown("### Detected Objects")
    
    # Count objects by class
    class_counts = {}
    for obj in objects:
        class_name = obj['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Display as metrics
    cols = st.columns(min(len(class_counts), 4))
    for i, (class_name, count) in enumerate(list(class_counts.items())[:4]):
        with cols[i]:
            st.metric(class_name.capitalize(), count)
    
    # Display details in expander
    with st.expander("View Details"):
        for obj in objects:
            conf = obj['confidence']
            st.markdown(f"- **{obj['class_name']}**: {conf:.1%} confidence")


def create_sidebar():
    """Create sidebar with theme toggle and configuration."""
    with st.sidebar:
        st.markdown("## Settings")
        
        # Theme toggle
        dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        st.divider()
        
        # Analysis configuration
        st.markdown("### Analysis Options")
        
        config = {}
        
        # Object detection
        config['object_detection'] = {
            'enabled': st.checkbox("Object Detection", value=True),
            'confidence_threshold': st.slider("Confidence", 0.1, 1.0, 0.5, 0.05),
        }
        
        # Face detection
        config['face_detection'] = {
            'enabled': st.checkbox("Face Detection", value=True)
        }
        
        # Color analysis
        config['color_extraction'] = {
            'enabled': st.checkbox("Color Analysis", value=True),
            'num_colors': st.slider("Number of Colors", 3, 10, 5)
        }
        
        # Text extraction
        config['text_extraction'] = {
            'enabled': st.checkbox("Text Recognition (OCR)", value=True)
        }
        
        # Image quality
        config['quality_analysis'] = {
            'enabled': st.checkbox("Quality Analysis", value=True)
        }
        
        st.divider()
        
        # Image history
        if st.session_state.image_history:
            st.markdown("### Recent Images")
            for i, item in enumerate(st.session_state.image_history):
                with st.container():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(item['thumbnail'], width=50)
                    with col2:
                        st.caption(item['timestamp'])
        
        return config


def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Apply theme
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">Multimodal AI System</h1>', unsafe_allow_html=True)
    
    # Feature badges
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1em;">
        <span class="feature-badge">YOLOv8 Detection</span>
        <span class="feature-badge">BLIP Captioning</span>
        <span class="feature-badge">Face Detection</span>
        <span class="feature-badge">Color Analysis</span>
        <span class="feature-badge">OCR</span>
        <span class="feature-badge">Quality Scoring</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Load system
    system = load_multimodal_system()
    if system is None:
        st.error("Failed to initialize the multimodal AI system. Please check the logs.")
        return
    
    # Sidebar configuration
    config = create_sidebar()
    
    # Main content
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Input tabs
    upload_tab, url_tab = st.tabs(["Upload Image", "Load from URL"])
    
    image = None
    temp_image_path = None
    
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, WebP"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    
    with url_tab:
        url = st.text_input("Enter image URL", placeholder="https://example.com/image.jpg")
        if url:
            with st.spinner("Loading image from URL..."):
                loaded_image = load_image_from_url(url)
                if loaded_image is not None:
                    image = Image.fromarray(cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB))
                else:
                    st.error("Failed to load image from URL")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Input Image")
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Ask a Question")
            
            # Question input
            question = st.text_input(
                "Your question",
                value=st.session_state.selected_question,
                placeholder="What do you see in this image?"
            )
            
            # Suggested questions (shown after first analysis)
            if st.session_state.current_results:
                suggestions = get_suggested_questions(st.session_state.current_results)
                st.markdown("**Suggested questions:**")
                suggestion_cols = st.columns(3)
                for i, suggestion in enumerate(suggestions[:3]):
                    with suggestion_cols[i]:
                        if st.button(suggestion[:30] + "..." if len(suggestion) > 30 else suggestion, key=f"sugg_{i}"):
                            st.session_state.selected_question = suggestion
                            st.rerun()
            
            analyze_button = st.button("Analyze Image", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if analyze_button and question:
            try:
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    temp_image_path = tmp_file.name
                
                # Progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Loading image...")
                progress_bar.progress(10)
                
                # Load image for additional processing
                cv_image = cv2.imread(temp_image_path)
                
                status_text.text("Running object detection...")
                progress_bar.progress(30)
                
                # Run main analysis
                results = system.process(temp_image_path, question, config)
                
                status_text.text("Detecting faces...")
                progress_bar.progress(50)
                
                # Face detection
                faces = []
                if config.get('face_detection', {}).get('enabled', True):
                    faces = detect_faces(cv_image)
                    results['faces'] = faces
                
                status_text.text("Analyzing image quality...")
                progress_bar.progress(70)
                
                # Quality analysis
                quality = None
                if config.get('quality_analysis', {}).get('enabled', True):
                    quality = calculate_image_quality(cv_image)
                    results['quality'] = quality
                
                status_text.text("Generating response...")
                progress_bar.progress(90)
                
                # Store results
                st.session_state.current_results = results
                
                # Add to history
                thumbnail = image.copy()
                thumbnail.thumbnail((100, 100))
                add_to_history(temp_image_path, thumbnail, results)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Display results
                st.divider()
                
                # AI Response
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### AI Response")
                
                if results.get('caption', {}).get('caption'):
                    st.markdown(f"**Caption:** {results['caption']['caption']}")
                
                if results.get('answer', {}).get('answer'):
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>Answer:</strong> {results['answer']['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Results tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Objects", "Faces", "Colors", "Text", "Quality"
                ])
                
                with tab1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    objects = results.get('features', {}).get('objects', [])
                    display_object_summary(objects)
                    
                    if objects:
                        # Show annotated image
                        annotated = display_image_with_detections(cv_image, objects, faces)
                        st.image(convert_bgr_to_rgb(annotated), caption="Detected Objects", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("### Face Detection")
                    if faces:
                        st.metric("Faces Detected", len(faces))
                        for i, face in enumerate(faces):
                            st.markdown(f"- Face {i+1}: {face['width']}x{face['height']} pixels")
                    else:
                        st.info("No faces detected in the image.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab3:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    colors = results.get('features', {}).get('colors', [])
                    display_color_palette(colors)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab4:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("### Extracted Text (OCR)")
                    ocr_result = results.get('features', {}).get('ocr_text', {})
                    text = ocr_result.get('text', '')
                    if text and text.strip():
                        st.code(text)
                    else:
                        st.info("No text detected in the image.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab5:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("### Image Quality Analysis")
                    if quality:
                        display_quality_metrics(quality)
                    else:
                        st.info("Quality analysis not available.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Download results
                st.divider()
                results_json = json.dumps(results, indent=2, cls=NumpyEncoder)
                st.download_button(
                    label="Download Analysis Results (JSON)",
                    data=results_json,
                    file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                # Cleanup
                if temp_image_path and os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">Error during analysis: {str(e)}</div>', 
                           unsafe_allow_html=True)
                logger.error(f"Analysis error: {str(e)}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: var(--text-secondary); padding: 1em;">
        Built with OpenCV, YOLOv8, BLIP, and Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
