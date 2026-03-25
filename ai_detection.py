"""
AI-Generated Image Detection Module
Uses EfficientNet-B4 model to detect AI-generated images (GAN, Diffusion)
"""

import os
import numpy as np
from PIL import Image
from keras.models import load_model

# Configuration
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Real', 'GAN', 'Diffusion']
MODEL_PATH = 'models/ai_generated/ai_detector_best.keras'

# Global model variable
_model = None

def load_model_if_needed():
    """Load the AI detection model if not already loaded"""
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(f"AI detection model not found at {MODEL_PATH}")
    return _model

def load_and_preprocess_image(image_path, img_size=IMG_SIZE):
    """Load and preprocess an image for the model."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        raise Exception(f"Error loading image: {e}")

def analyze_ai_generated(image_path):
    """
    Analyze an image for AI-generation using EfficientNet-B4 model.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary with prediction results:
        - predicted_class: 'Real', 'GAN', or 'Diffusion'
        - confidence: Confidence score (0-1)
        - probabilities: Dictionary with probabilities for each class
    """
    model = load_model_if_needed()
    
    # Load and preprocess image
    img_array = load_and_preprocess_image(image_path)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_batch, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx])
    
    # Get probabilities for all classes
    probabilities = {
        CLASS_NAMES[i]: float(predictions[0][i]) 
        for i in range(len(CLASS_NAMES))
    }
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities
    }

