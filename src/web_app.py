"""
Enhanced Streamlit web application for the Multimodal AI System.

Features:
- Client-Server Architecture (connects to FastAPI backend)
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
import time
import requests
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import convert_bgr_to_rgb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_URL = "http://localhost:8000"

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
    if 'api_healthy' not in st.session_state:
        st.session_state.api_healthy = False


def check_api_health() -> bool:
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


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


def add_to_history(image_path_or_url: str, thumbnail: Image.Image, results: Dict):
    """Add analyzed image to history."""
    history_item = {
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'image_path': image_path_or_url,
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
    
    features = results.get('features', {})
    
    # Based on detected objects
    if features.get('objects'):
        objects = features['objects']
        if objects:
            class_names = list(set([obj['class_name'] for obj in objects]))
            if len(class_names) == 1:
                questions.append(f"What is the {class_names[0]} doing?")
            else:
                questions.append(f"How do the {class_names[0]} and {class_names[1]} relate?")
            questions.append("What is the main subject of this image?")
    
    # Based on colors
    if features.get('colors'):
        questions.append("Why are these colors dominant in this image?")
    
    # Based on text
    if features.get('ocr_text', {}).get('text'):
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
    
    # Draw face detections if available from backend (backend upgrade needed for this)
    # For now, we rely on the results structure
    
    return result_image


def display_quality_metrics(quality: Dict):
    """Display image quality metrics."""
    score = quality.get('overall_score', 0)
    
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
            <div class="stat-number">{quality.get('brightness', 0):.0f}</div>
            <div class="stat-label">Brightness</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{quality.get('contrast', 0):.0f}</div>
            <div class="stat-label">Contrast</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        blur_text = "Yes" if quality.get('is_blurry', False) else "No"
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
            percentage = color.get('percentage', 0)
            
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
        
        # Connection Status
        if st.button("Check API Connection"):
            if check_api_health():
                st.success("Connected to Backend API")
                st.session_state.api_healthy = True
            else:
                st.error("Cannot connect to Backend API")
                st.session_state.api_healthy = False
        
        if st.session_state.api_healthy:
            st.markdown('<div class="success-box">API Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">API Disconnected</div>', unsafe_allow_html=True)
            st.caption(f"Make sure API is running at {API_URL}")

        st.divider()
        
        # Analysis configuration
        st.markdown("### Analysis Options")
        
        config = {}

        # Model Selection (New 10x Feature)
        st.markdown("#### Model Backend")
        llm_model = st.selectbox(
            "Vision Language Model",
            ["blip", "llava"],
            format_func=lambda x: "LlaVA (Smart & Slow)" if x == "llava" else "BLIP (Fast & Basic)",
            help="Choose 'LlaVA' for complex reasoning (requires GPU). Choose 'BLIP' for speed."
        )
        config['llm_model'] = llm_model
        
        # Reasoning Mode
        reasoning_mode = st.radio(
            "Analysis Mode",
            ["detailed", "fast"],
            format_func=lambda x: "Detailed (Visual CoT)" if x == "detailed" else "Fast (Global Caption only)",
            help="Detailed mode crops objects and captions them individually for better reasoning."
        )
        config['reasoning_mode'] = reasoning_mode
        
        # Object detection
        config['object_detection'] = {
            'enabled': st.checkbox("Object Detection", value=True),
            'confidence_threshold': st.slider("Confidence", 0.1, 1.0, 0.5, 0.05),
        }
        
        # Face detection (Placeholder for now as it needs backend update to be fully configurable via config)
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
                        if st.button("Load", key=f"hist_{i}"):
                            st.session_state.current_results = item.get('results')
                            st.rerun()
        
        return config


def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Check health on first load
    if not st.session_state.api_healthy:
        if check_api_health():
           st.session_state.api_healthy = True
    
    # Apply theme
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">Multimodal AI System</h1>', unsafe_allow_html=True)
    
    # Feature badges
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1em;">
        <span class="feature-badge">Async API</span>
        <span class="feature-badge">YOLOv8 Detection</span>
        <span class="feature-badge">BLIP Captioning</span>
        <span class="feature-badge">Color Analysis</span>
        <span class="feature-badge">OCR</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    config = create_sidebar()
    
    # Main content
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Input tabs
    upload_tab, url_tab = st.tabs(["Upload Image", "Load from URL"])
    
    image = None
    image_bytes = None
    image_source = None
    
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, WebP"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()
            image_source = "upload"
    
    with url_tab:
        url = st.text_input("Enter image URL", placeholder="https://example.com/image.jpg")
        if url:
            with st.spinner("Loading image from URL..."):
                loaded_image = load_image_from_url(url)
                if loaded_image is not None:
                    # Convert to PIL for consistency
                    image = Image.fromarray(cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB))
                    # Encode back to bytes for API
                    success, encoded_img = cv2.imencode('.jpg', loaded_image)
                    if success:
                        image_bytes = encoded_img.tobytes()
                    image_source = "url"
                else:
                    st.error("Failed to load image from URL")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Question input
    question = st.text_input(
        "Ask a question about the image:", 
        value=st.session_state.selected_question if st.session_state.selected_question else "Describe this image in detail"
    )
    
    # Analysis button
    if image is not None and st.button("Analyze Image", type="primary"):
        if not st.session_state.api_healthy:
            st.error("API is not connected. Please make sure the backend server is running.")
        else:
            with st.spinner("Submitting analysis job..."):
                try:
                    # Prepare config
                    config_json = json.dumps(config)
                    
                    # Send request
                    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
                    data = {"question": question, "config": config_json}
                    
                    response = requests.post(f"{API_URL}/analyze", files=files, data=data) 
                    
                    if response.status_code == 200:
                        job_data = response.json()
                        job_id = job_data['job_id']
                        
                        # Polling loop
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        while True:
                            status_response = requests.get(f"{API_URL}/jobs/{job_id}")
                            if status_response.status_code != 200:
                                st.error("Failed to get job status")
                                break
                                
                            job_status = status_response.json()
                            status = job_status['status']
                            
                            if status == "PENDING":
                                status_text.text("Job pending...")
                                progress_bar.progress(20)
                            elif status == "PROCESSING":
                                status_text.text("Processing image (YOLO + BLIP + OCR)...")
                                progress_bar.progress(60)
                            elif status == "COMPLETED":
                                progress_bar.progress(100)
                                status_text.success("Analysis complete!")
                                st.session_state.current_results = job_status['result']
                                # Add thumbnail
                                thumbnail = image.copy()
                                thumbnail.thumbnail((100, 100))
                                add_to_history(image_source, thumbnail, job_status['result'])
                                break
                            elif status == "FAILED":
                                st.error(f"Analysis failed: {job_status.get('error')}")
                                break
                                
                            time.sleep(1) # Poll every second
                            
                    else:
                        st.error(f"Failed to submit job: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    
    # Display Results
    if st.session_state.current_results:
        results = st.session_state.current_results
        
        st.divider()
        
        # Featured Result: Answer and Caption
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Image")
            # Convert PIL image to array for OpenCV drawing
            if image:
                img_array = np.array(image)
                # Convert RGB to BGR for OpenCV
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Draw detections
                features = results.get('features', {})
                objects = features.get('objects', [])
                # Note: Faces logic needs to be propagated from backend features if relevant
                
                annotated_img = display_image_with_detections(img_array, objects)
                
                # Convert back to RGB for Streamlit
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            elif image_source:
                 st.info("Image not loaded in current context (loaded from history/url)")
        
        with col2:
            st.markdown("### AI Analysis")
            
            # Caption
            caption = results.get('caption', {}).get('caption', 'No caption generated')
            st.markdown(f"""
            <div class="info-box">
                <strong>📝 Caption:</strong><br>
                {caption}
            </div>
            """, unsafe_allow_html=True)
            
            # Answer
            answer = results.get('answer', {}).get('answer', 'No answer generated')
            st.markdown(f"""
            <div class="success-box">
                <strong>💡 Answer to "{results.get('question', 'your question')}":</strong><br>
                {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Key Findings
            st.markdown("#### Key Findings")
            summary = results.get('summary', {})
            for finding in summary.get('key_findings', []):
                st.markdown(f"- {finding}")
        
        # Tabs for details
        tab1, tab2, tab3, tab4 = st.tabs(["Objects", "Colors", "Text", "Stats"])
        
        with tab1:
            features = results.get('features', {})
            display_object_summary(features.get('objects', []))
            
        with tab2:
            features = results.get('features', {})
            display_color_palette(features.get('colors', []))
            
        with tab3:
            features = results.get('features', {})
            ocr_text = features.get('ocr_text', {}).get('text', '')
            if ocr_text:
                st.markdown("### Extracted Text")
                st.text_area("OCR Result", ocr_text, height=150)
            else:
                st.info("No text detected.")
                
        with tab4:
             features = results.get('features', {})
             stats = features.get('image_stats', {})
             # Using the image stats which might contain the quality metrics
             # Ideally, we should ensure the backend fills this.
             # For now, we display what we have.
             st.json(stats)
        
        # Suggested questions
        st.divider()
        st.markdown("### Follow-up Questions")
        
        suggestions = get_suggested_questions(results)
        
        cols = st.columns(3)
        for i, q in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(q, key=f"q_{i}"):
                    st.session_state.selected_question = q
                    st.rerun()

if __name__ == "__main__":
    main()
