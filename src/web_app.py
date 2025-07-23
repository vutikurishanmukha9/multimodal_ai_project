"""
Streamlit web application for the Multimodal AI System.

This module provides a user-friendly web interface for:
- Image upload and display
- Interactive question asking
- Visual results presentation
- Analysis configuration
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
import logging
from typing import Dict, Any, Optional
import tempfile
import os

# Import our modules
from .multimodal_system import MultimodalAI
from .utils import convert_bgr_to_rgb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multimodal AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1em;
    }
    .subheader {
        font-size: 1.2em;
        color: #333;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1em;
        border-radius: 0.5em;
        margin: 1em 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1em;
        border-radius: 0.5em;
        margin: 1em 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1em;
        border-radius: 0.5em;
        margin: 1em 0;
    }
</style>
""", unsafe_allow_html=True)


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
        st.error(f"Failed to load multimodal system: {str(e)}")
        return None


def display_image_with_detections(image_path: str, detections: list) -> None:
    """Display image with object detection overlays."""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            st.error("Failed to load image")
            return

        # Draw detections
        if detections:
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
            ]

            for i, detection in enumerate(detections):
                color = colors[i % len(colors)]
                x1, y1, x2, y2 = detection['bbox']
                label = f"{detection['class_name']}: {detection['confidence']:.2f}"

                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(image, (x1, y1 - text_h - baseline - 5), 
                             (x1 + text_w, y1), color, -1)

                # Draw text
                cv2.putText(image, label, (x1, y1 - baseline - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Convert BGR to RGB for display
        image_rgb = convert_bgr_to_rgb(image)
        st.image(image_rgb, caption="Image with Object Detections", use_column_width=True)

    except Exception as e:
        st.error(f"Error displaying image with detections: {str(e)}")


def display_color_palette(colors: list) -> None:
    """Display extracted colors as a palette."""
    if not colors:
        return

    st.markdown("### 🎨 Dominant Colors")

    # Create columns for color display
    cols = st.columns(min(len(colors), 5))

    for i, color in enumerate(colors[:5]):
        with cols[i]:
            # Create color swatch
            color_rgb = color['rgb']
            hex_color = color['hex']
            percentage = color['percentage']

            # Display color rectangle using HTML/CSS
            st.markdown(f"""
                <div style="
                    background-color: {hex_color};
                    height: 80px;
                    border-radius: 10px;
                    border: 2px solid #ddd;
                    margin-bottom: 5px;
                "></div>
                <p style="text-align: center; margin: 0;">
                    <strong>{hex_color}</strong><br>
                    {percentage:.1f}%
                </p>
            """, unsafe_allow_html=True)


def display_object_summary(objects: list) -> None:
    """Display object detection summary."""
    if not objects:
        st.info("No objects detected in the image.")
        return

    st.markdown("### 🔍 Detected Objects")

    # Count objects by class
    object_counts = {}
    total_confidence = 0

    for obj in objects:
        class_name = obj['class_name']
        confidence = obj['confidence']

        if class_name not in object_counts:
            object_counts[class_name] = {'count': 0, 'confidences': []}

        object_counts[class_name]['count'] += 1
        object_counts[class_name]['confidences'].append(confidence)
        total_confidence += confidence

    # Display summary statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Objects", len(objects))

    with col2:
        st.metric("Unique Classes", len(object_counts))

    with col3:
        avg_confidence = total_confidence / len(objects) if objects else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")

    # Display detailed breakdown
    st.markdown("#### Object Breakdown")
    for class_name, data in object_counts.items():
        count = data['count']
        avg_conf = np.mean(data['confidences'])
        st.write(f"• **{class_name}**: {count} detected (avg confidence: {avg_conf:.2f})")


def display_ocr_results(ocr_data: dict) -> None:
    """Display OCR text extraction results."""
    text = ocr_data.get('text', '').strip()
    confidence = ocr_data.get('confidence', 0)
    word_count = ocr_data.get('word_count', 0)

    st.markdown("### 📝 Text Recognition (OCR)")

    if text:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.text_area("Extracted Text", text, height=100, disabled=True)

        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")
            st.metric("Word Count", word_count)
    else:
        st.info("No text detected in the image.")


def create_sidebar_config() -> Dict[str, Any]:
    """Create configuration sidebar."""
    st.sidebar.markdown("## ⚙️ Analysis Configuration")

    config = {}

    # Image preprocessing
    st.sidebar.markdown("### Image Preprocessing")
    config['target_size'] = st.sidebar.selectbox(
        "Target Size", 
        [(640, 640), (512, 512), (416, 416), (320, 320)],
        index=0,
        format_func=lambda x: f"{x[0]}x{x[1]}"
    )
    config['normalize'] = st.sidebar.checkbox("Normalize Images", value=True)

    # Object detection
    st.sidebar.markdown("### Object Detection")
    obj_config = {}
    obj_config['enabled'] = st.sidebar.checkbox("Enable Object Detection", value=True)
    obj_config['confidence_threshold'] = st.sidebar.slider(
        "Confidence Threshold", 0.1, 1.0, 0.5, 0.05
    )
    obj_config['iou_threshold'] = st.sidebar.slider(
        "IoU Threshold", 0.1, 1.0, 0.45, 0.05
    )
    config['object_detection'] = obj_config

    # Color extraction
    st.sidebar.markdown("### Color Analysis")
    color_config = {}
    color_config['enabled'] = st.sidebar.checkbox("Enable Color Analysis", value=True)
    color_config['num_colors'] = st.sidebar.slider("Number of Colors", 3, 10, 5)
    color_config['method'] = st.sidebar.selectbox("Method", ["kmeans", "histogram"])
    config['color_extraction'] = color_config

    # Text extraction
    st.sidebar.markdown("### Text Recognition")
    ocr_config = {}
    ocr_config['enabled'] = st.sidebar.checkbox("Enable OCR", value=True)
    config['text_extraction'] = ocr_config

    # Language model
    st.sidebar.markdown("### Language Generation")
    caption_config = {}
    caption_config['max_length'] = st.sidebar.slider("Caption Max Length", 20, 100, 50)
    caption_config['num_beams'] = st.sidebar.slider("Beam Search Beams", 1, 10, 5)
    caption_config['temperature'] = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    config['captioning'] = caption_config

    qa_config = {}
    qa_config['max_length'] = st.sidebar.slider("Answer Max Length", 50, 200, 100)
    qa_config['use_features_context'] = st.sidebar.checkbox(
        "Use Features in Context", value=True
    )
    config['question_answering'] = qa_config

    return config


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🤖 Multimodal AI System</h1>', 
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <p>Upload an image and ask questions about it! This system combines computer vision and natural language processing to analyze images comprehensively.</p>
        <p><strong>Features:</strong> Object Detection (YOLOv8) • Image Captioning (BLIP) • Color Analysis • Text Recognition (OCR) • Visual Question Answering</p>
    </div>
    """, unsafe_allow_html=True)

    # Load the system
    system = load_multimodal_system()
    if system is None:
        st.error("Failed to initialize the multimodal AI system. Please check the logs.")
        return

    # Create sidebar configuration
    config = create_sidebar_config()

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, WebP"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                temp_image_path = tmp_file.name

    with col2:
        st.markdown("### ❓ Ask a Question")
        question = st.text_input(
            "What would you like to know about the image?",
            placeholder="e.g., What objects do you see in this image?",
            help="Ask any question about the image content, objects, colors, or scene."
        )

        analyze_button = st.button("🔍 Analyze Image", type="primary")

    # Analysis section
    if uploaded_file is not None and question and analyze_button:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # Show progress
        with st.spinner("Analyzing image... This may take a moment."):
            try:
                # Run analysis
                results = system.process(temp_image_path, question, config)

                if 'error' in results:
                    st.markdown(f'<div class="error-box">❌ {results["error"]}</div>', 
                               unsafe_allow_html=True)
                else:
                    # Display results in tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "🤖 AI Response", "🔍 Objects", "🎨 Colors", "📝 Text", "📈 Summary"
                    ])

                    with tab1:
                        st.markdown("### 💬 Caption & Answer")

                        # Display caption
                        caption_data = results.get('caption', {})
                        caption = caption_data.get('caption', '')
                        if caption:
                            st.markdown(f"**Image Caption:** {caption}")
                            st.caption(f"Confidence: {caption_data.get('confidence', 0):.2f}")

                        # Display answer
                        answer_data = results.get('answer', {})
                        answer = answer_data.get('answer', '')
                        if answer:
                            st.markdown(f"**Answer to '{question}':**")
                            st.info(answer)
                            st.caption(f"Confidence: {answer_data.get('confidence', 0):.2f}")

                    with tab2:
                        features = results.get('features', {})
                        objects = features.get('objects', [])

                        if objects:
                            display_object_summary(objects)
                            display_image_with_detections(temp_image_path, objects)
                        else:
                            st.info("No objects detected in the image.")

                    with tab3:
                        features = results.get('features', {})
                        colors = features.get('colors', [])
                        display_color_palette(colors)

                    with tab4:
                        features = results.get('features', {})
                        ocr_data = features.get('ocr_text', {})
                        display_ocr_results(ocr_data)

                    with tab5:
                        st.markdown("### 📊 Analysis Summary")
                        summary = results.get('summary', {})

                        if summary:
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "Objects Detected", 
                                    summary.get('features_summary', {}).get('objects_detected', 0)
                                )

                            with col2:
                                st.metric(
                                    "Colors Extracted", 
                                    summary.get('features_summary', {}).get('dominant_colors_extracted', 0)
                                )

                            with col3:
                                text_detected = summary.get('features_summary', {}).get('text_detected', False)
                                st.metric("Text Detected", "Yes" if text_detected else "No")

                            # Key findings
                            key_findings = summary.get('key_findings', [])
                            if key_findings:
                                st.markdown("#### 🔑 Key Findings")
                                for finding in key_findings:
                                    st.write(f"• {finding}")

                        # Download results
                        st.markdown("#### 💾 Download Results")
                        results_json = json.dumps(results, indent=2)
                        st.download_button(
                            label="📄 Download Analysis Results (JSON)",
                            data=results_json,
                            file_name=f"analysis_results_{uploaded_file.name}.json",
                            mime="application/json"
                        )

                # Clean up temporary file
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)

            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error during analysis: {str(e)}</div>', 
                           unsafe_allow_html=True)
                logger.error(f"Analysis error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        <p>Multimodal AI System v1.0 | Built with Streamlit, OpenCV, YOLOv8, and BLIP</p>
        <p>For best results, use clear, well-lit images with visible objects and text.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
