#!/bin/bash
# Quick start script for Fake Image Detector Web App

echo "🔍 Starting Fake Image Detector Web Application..."
echo ""

# Check if models exist
if [ ! -f "models/ai_generated/ai_detector_best.keras" ]; then
    echo "⚠️  Warning: AI detection model not found!"
    echo "   Expected: models/ai_generated/ai_detector_best.keras"
fi

if [ ! -f "models/fake_image_detector_model.keras" ]; then
    echo "⚠️  Warning: ELA detection model not found!"
    echo "   Expected: models/fake_image_detector_model.keras"
fi

echo "📦 Checking dependencies..."
python3 -c "import streamlit, keras, PIL, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
else
    echo "✅ Dependencies OK"
fi

echo ""
echo "🚀 Launching Streamlit app..."
echo "   The app will open in your browser at http://localhost:8501"
echo ""

streamlit run app.py

