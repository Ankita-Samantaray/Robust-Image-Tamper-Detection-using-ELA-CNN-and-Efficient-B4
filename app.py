#!/usr/bin/env python3
"""
Fake Image Detector Web Application
Integrates three detection methods:
1. AI-Generated Image Detection (EfficientNet-B4)
2. CNN-ELA Detection
3. Metadata Analysis
"""

import streamlit as st
import os
import numpy as np
from PIL import Image
import tempfile
from datetime import datetime

# Import analysis modules
from ai_detection import analyze_ai_generated
from ela_detection import analyze_ela
from metadata_detection import analyze_metadata

# Page configuration
st.set_page_config(
    page_title="Fake Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .analysis-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .real-result {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .fake-result {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .warning-result {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake Image Detector</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Comprehensive image authenticity analysis using multiple detection methods
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This tool analyzes images using three methods:
        
        1. **AI-Generated Detection**
           - Uses EfficientNet-B4
           - Detects GAN and Diffusion model images
        
        2. **CNN-ELA Detection**
           - Error Level Analysis + CNN
           - Detects image manipulation
        
        3. **Metadata Analysis**
           - EXIF data analysis
           - Detects tampering indicators
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Status")
        
        # Check if models exist
        ai_model_exists = os.path.exists('models/ai_generated/ai_detector_best.keras')
        ela_model_exists = os.path.exists('models/fake_image_detector_model.keras')
        
        if ai_model_exists:
            st.success("✓ AI Detection Model")
        else:
            st.error("✗ AI Detection Model")
            
        if ela_model_exists:
            st.success("✓ ELA Detection Model")
        else:
            st.error("✗ ELA Detection Model")
            
        st.info("✓ Metadata Analysis")
    
    # Main content
    st.header("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Save uploaded file temporarily
        tmp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_path = tmp_file.name
        
        # Save the image (convert if needed)
        try:
            if uploaded_file.type in ['image/png', 'image/bmp'] or image.format in ['PNG', 'BMP']:
                # Convert to RGB if needed and save as JPEG
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(tmp_path, 'JPEG', quality=95)
            else:
                # For JPEG files, write directly
                uploaded_file.seek(0)  # Reset file pointer
                with open(tmp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
        except Exception as e:
            st.error(f"Error saving image: {str(e)}")
            tmp_path = None
        
        # Analyze button (only enable if file was saved successfully)
        if tmp_path and os.path.exists(tmp_path):
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image... This may take a few moments."):
                    # Run all three analyses
                    results = {}
                    
                    # 1. AI-Generated Detection
                    try:
                        st.subheader("🤖 AI-Generated Image Detection")
                        ai_results = analyze_ai_generated(tmp_path)
                        results['ai'] = ai_results
                        display_ai_results(ai_results)
                    except Exception as e:
                        st.error(f"AI Detection Error: {str(e)}")
                        results['ai'] = None
                    
                    st.markdown("---")
                    
                    # 2. CNN-ELA Detection
                    try:
                        st.subheader("🖼️ CNN-ELA Detection")
                        ela_results = analyze_ela(tmp_path)
                        results['ela'] = ela_results
                        display_ela_results(ela_results)
                    except Exception as e:
                        st.error(f"ELA Detection Error: {str(e)}")
                        results['ela'] = None
                    
                    st.markdown("---")
                    
                    # 3. Metadata Analysis
                    try:
                        st.subheader("📋 Metadata Analysis")
                        metadata_results = analyze_metadata(tmp_path)
                        results['metadata'] = metadata_results
                        display_metadata_results(metadata_results)
                    except Exception as e:
                        st.error(f"Metadata Analysis Error: {str(e)}")
                        results['metadata'] = None
                    
                    st.markdown("---")
                    
                    # Final Summary
                    st.subheader("📊 Final Summary")
                    display_final_summary(results)
        else:
            st.error("Failed to save uploaded image. Please try again.")
        
        # Cleanup
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass  # Ignore cleanup errors
    
    else:
        st.info("👆 Please upload an image to begin analysis")
        
        # Show example
        st.markdown("---")
        st.subheader("📝 How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1. AI Detection**
            - Analyzes image features
            - Classifies as Real/GAN/Diffusion
            - Provides confidence scores
            """)
        
        with col2:
            st.markdown("""
            **2. ELA Detection**
            - Performs Error Level Analysis
            - Uses CNN to detect manipulation
            - Identifies edited regions
            """)
        
        with col3:
            st.markdown("""
            **3. Metadata Analysis**
            - Extracts EXIF data
            - Checks for inconsistencies
            - Identifies tampering indicators
            """)


def display_ai_results(results):
    """Display AI-generated detection results"""
    if results is None:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        predicted_class = results['predicted_class']
        confidence = results['confidence']
        probabilities = results['probabilities']
        
        # Determine result box style
        if predicted_class == 'Real':
            box_class = "real-result"
            icon = "✅"
        else:
            box_class = "fake-result"
            icon = "⚠️"
        
        st.markdown(f"""
        <div class="result-box {box_class}">
            <h3>{icon} Prediction: {predicted_class}</h3>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Probability Distribution")
        # Create probability bar chart
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        
        # Create a simple bar chart using st.bar_chart
        prob_dict = {k: v for k, v in zip(classes, probs)}
        st.bar_chart(prob_dict)
    
    # Detailed probabilities
    with st.expander("📊 Detailed Probabilities"):
        for class_name, prob in probabilities.items():
            st.progress(prob, text=f"{class_name}: {prob:.2%}")


def display_ela_results(results):
    """Display CNN-ELA detection results"""
    if results is None:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        prediction = results['prediction']
        confidence = results['confidence']
        
        # Determine result box style
        if prediction == 'Real':
            box_class = "real-result"
            icon = "✅"
        else:
            box_class = "fake-result"
            icon = "⚠️"
        
        st.markdown(f"""
        <div class="result-box {box_class}">
            <h3>{icon} Prediction: {prediction}</h3>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Method")
        st.info("""
        **Error Level Analysis (ELA) + CNN**
        
        - Converts image to ELA format
        - Analyzes compression artifacts
        - Uses CNN to detect manipulation patterns
        """)


def display_metadata_results(results):
    """Display metadata analysis results"""
    if results is None:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        tampering_score = results['tampering_score']
        likely_tampered = results['analysis_summary']['likely_tampered']
        confidence = results['analysis_summary']['tampering_confidence']
        
        # Determine result box style
        if likely_tampered:
            box_class = "fake-result"
            icon = "⚠️"
        elif tampering_score > 30:
            box_class = "warning-result"
            icon = "🔍"
        else:
            box_class = "real-result"
            icon = "✅"
        
        st.markdown(f"""
        <div class="result-box {box_class}">
            <h3>{icon} Tampering Assessment</h3>
            <p><strong>Score:</strong> {tampering_score}/100</p>
            <p><strong>Likely Tampered:</strong> {'Yes' if likely_tampered else 'No'}</p>
            <p><strong>Confidence:</strong> {confidence}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Metadata Quality")
        metadata_completeness = results['metadata_completeness']
        has_exif = results['has_exif']
        
        st.metric("Completeness", f"{metadata_completeness:.1f}%")
        st.metric("Has EXIF Data", "Yes" if has_exif else "No")
    
    # Indicators
    if results['tampering_indicators']:
        with st.expander("🔍 Tampering Indicators"):
            for i, indicator in enumerate(results['tampering_indicators'], 1):
                st.write(f"{i}. {indicator}")
    
    # Key EXIF information
    with st.expander("📋 Key EXIF Information"):
        exif_data = results['exif_data']
        key_fields = ['Make', 'Model', 'Software', 'DateTime', 'DateTimeOriginal']
        
        for field in key_fields:
            if field in exif_data:
                st.write(f"**{field}:** {exif_data[field]}")


def display_final_summary(results):
    """Display final comprehensive summary"""
    summary_data = []
    
    # AI Detection Summary
    if results.get('ai'):
        ai = results['ai']
        summary_data.append({
            'Method': 'AI-Generated Detection',
            'Result': ai['predicted_class'],
            'Confidence': f"{ai['confidence']:.2%}",
            'Status': '✅ Real' if ai['predicted_class'] == 'Real' else '⚠️ AI-Generated'
        })
    
    # ELA Detection Summary
    if results.get('ela'):
        ela = results['ela']
        summary_data.append({
            'Method': 'CNN-ELA Detection',
            'Result': ela['prediction'],
            'Confidence': f"{ela['confidence']:.2%}",
            'Status': '✅ Real' if ela['prediction'] == 'Real' else '⚠️ Fake'
        })
    
    # Metadata Summary
    if results.get('metadata'):
        meta = results['metadata']
        likely_tampered = meta['analysis_summary']['likely_tampered']
        summary_data.append({
            'Method': 'Metadata Analysis',
            'Result': 'Likely Tampered' if likely_tampered else 'No Tampering',
            'Confidence': meta['analysis_summary']['tampering_confidence'],
            'Status': '⚠️ Suspicious' if likely_tampered else '✅ Clean'
        })
    
    # Display summary table
    if summary_data:
        import pandas as pd
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Overall verdict
        st.markdown("### 🎯 Overall Verdict")
        
        # Count votes
        real_votes = sum(1 for r in summary_data if 'Real' in r['Result'] or 'No Tampering' in r['Result'] or 'Clean' in r['Status'])
        suspicious_votes = len(summary_data) - real_votes
        
        if real_votes > suspicious_votes:
            st.success("""
            **✅ Image appears to be AUTHENTIC**
            
            Based on the analysis of multiple detection methods, this image shows characteristics 
            consistent with an authentic, non-manipulated photograph.
            """)
        elif suspicious_votes > real_votes:
            st.error("""
            **⚠️ Image appears to be FAKE or MANIPULATED**
            
            Multiple detection methods indicate this image may be AI-generated or has been 
            manipulated. Please verify the source and authenticity.
            """)
        else:
            st.warning("""
            **🔍 INCONCLUSIVE RESULTS**
            
            The analysis shows mixed results. The image may be authentic but processed, or 
            it may be a sophisticated fake. Additional verification is recommended.
            """)


if __name__ == "__main__":
    main()

