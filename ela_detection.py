"""
CNN-ELA Detection Module
Uses Error Level Analysis (ELA) + CNN to detect image manipulation
"""

import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from keras.models import load_model

# Configuration
IMG_SIZE = (128, 128)
MODEL_PATH = 'models/fake_image_detector_model.keras'

# Global model variable
_model = None

def load_model_if_needed():
    """Load the ELA detection model if not already loaded"""
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(f"ELA detection model not found at {MODEL_PATH}")
    return _model

def convert_to_ela_image(path, quality=90):
    """
    Convert image to Error Level Analysis (ELA) format.
    
    Args:
        path: Path to image file
        quality: JPEG quality for resaving
    
    Returns:
        PIL Image object in ELA format
    """
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    
    ela_im = ImageChops.difference(im, resaved_im)
    
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    
    # Clean up temporary file
    if os.path.exists(resaved_filename):
        os.remove(resaved_filename)
    
    return ela_im

def analyze_ela(image_path):
    """
    Analyze an image using CNN-ELA method.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary with prediction results:
        - prediction: 'Real' or 'Fake'
        - confidence: Confidence score (0-1)
    """
    model = load_model_if_needed()
    
    # Convert to ELA and preprocess
    ela_im = convert_to_ela_image(image_path, quality=90)
    ela_im = ela_im.resize(IMG_SIZE)
    
    # Convert to array and normalize
    feature = np.array(ela_im).flatten() / 255.0
    feature = feature.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3)
    
    # Predict
    predicted_array = model.predict(feature, verbose=0)
    
    # Determine prediction
    if predicted_array[0][0] > predicted_array[0][1]:
        prediction = 'Real'
        confidence = float(predicted_array[0][0])
    else:
        prediction = 'Fake'
        confidence = float(predicted_array[0][1])
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': {
            'Real': float(predicted_array[0][0]),
            'Fake': float(predicted_array[0][1])
        }
    }

