# Fake Image Detector Web Application

A comprehensive web application that analyzes images using three different detection methods to determine authenticity.

## Features

1. **AI-Generated Image Detection**
   - Uses EfficientNet-B4 deep learning model
   - Detects GAN-generated and Diffusion model images
   - Classifies images as Real, GAN, or Diffusion
   - Provides confidence scores for each class

2. **CNN-ELA Detection**
   - Error Level Analysis (ELA) preprocessing
   - CNN-based manipulation detection
   - Identifies edited or tampered regions
   - Binary classification: Real or Fake

3. **Metadata Analysis**
   - EXIF data extraction and analysis
   - Timestamp inconsistency detection
   - Editing software detection
   - Tampering score calculation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure model files are present:
   - `models/ai_generated/ai_detector_best.keras`
   - `models/fake_image_detector_model.keras`

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## How to Use

1. **Upload Image**: Click "Browse files" and select an image (JPG, JPEG, PNG, BMP)

2. **Analyze**: Click the "Analyze Image" button

3. **Review Results**: 
   - View results from each detection method
   - Check the final summary for overall verdict
   - Expand sections for detailed information

## Output

The application provides:
- **Individual Analysis Results**: Results from each detection method
- **Confidence Scores**: Probability scores for each classification
- **Detailed Probabilities**: Breakdown of all class probabilities
- **Tampering Indicators**: List of suspicious findings
- **Final Summary**: Overall verdict based on all methods

## File Structure

```
.
├── app.py                 # Main Streamlit application
├── ai_detection.py        # AI-generated detection module
├── ela_detection.py       # CNN-ELA detection module
├── metadata_detection.py  # Metadata analysis module
├── requirements.txt       # Python dependencies
└── models/               # Trained model files
    ├── ai_generated/
    │   └── ai_detector_best.keras
    └── fake_image_detector_model.keras
```

## Notes

- The application requires trained models to function
- First run may take longer as models are loaded into memory
- Large images are automatically resized for processing
- Temporary files are cleaned up after analysis

