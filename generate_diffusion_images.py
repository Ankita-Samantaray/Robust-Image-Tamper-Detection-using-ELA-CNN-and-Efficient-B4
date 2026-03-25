#!/usr/bin/env python3
"""
Generate diffusion model images using Stable Diffusion.
Saves images to datasets/ai_detection/diffusion/
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

try:
    from diffusers import StableDiffusionPipeline
    import torch
    from PIL import Image
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Please install: pip install diffusers transformers accelerate torch pillow")
    sys.exit(1)

# Configuration
OUTPUT_DIR = "datasets/ai_detection/diffusion"
NUM_IMAGES = 500  # Adjust this number as needed (started with 500 for testing)
BATCH_SIZE = 1  # Generate one at a time to avoid memory issues

# Diverse prompts for variety
PROMPTS = [
    # Landscapes
    "a beautiful mountain landscape with snow peaks and a lake in the foreground",
    "a serene beach at sunset with palm trees and calm ocean waves",
    "a dense forest with sunlight filtering through the trees",
    "a vast desert with sand dunes under a clear blue sky",
    "a tropical island with crystal clear water and white sandy beaches",
    
    # Portraits
    "a professional portrait of a person with natural lighting",
    "a close-up portrait of a person in formal attire",
    "a portrait of a person in casual clothing, outdoor setting",
    
    # Urban scenes
    "a bustling city street with modern buildings and cars",
    "a quiet urban park with benches and walking paths",
    "a city skyline at night with illuminated buildings",
    "a vintage street with old buildings and cobblestone",
    
    # Nature
    "a field of wildflowers in bloom with mountains in background",
    "a waterfall in a tropical rainforest",
    "a frozen lake surrounded by pine trees in winter",
    "a garden with colorful flowers and butterflies",
    
    # Objects
    "a still life of fruits arranged on a wooden table",
    "a vintage car parked on a street",
    "a cozy interior room with furniture and plants",
    "a modern kitchen with stainless steel appliances",
    
    # Abstract/Artistic
    "an abstract art piece with vibrant colors and geometric shapes",
    "a surreal landscape with floating islands and unusual lighting",
    "a fantasy scene with magical elements and ethereal atmosphere",
    
    # Animals
    "a cat sitting on a windowsill looking outside",
    "a dog playing in a park with green grass",
    "birds flying over a landscape at golden hour",
    
    # Architecture
    "a modern building with glass facades reflecting the sky",
    "an ancient temple with intricate carvings and stone pillars",
    "a bridge spanning across a river with city in background",
    
    # Everyday scenes
    "people walking on a busy street in a city",
    "a coffee shop interior with people and plants",
    "a library with bookshelves and reading areas",
]

def generate_images():
    """Generate diffusion images using Stable Diffusion."""
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {NUM_IMAGES} diffusion images...")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Using Stable Diffusion v1.5")
    print()
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("Warning: Using CPU will be slow. Consider using GPU for faster generation.")
    print()
    
    # Load the pipeline
    print("Loading Stable Diffusion model (this may take a few minutes on first run)...")
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for faster generation
            requires_safety_checker=False
        )
        pipeline = pipeline.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except:
                pass
                
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative model...")
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32
            )
            pipeline = pipeline.to(device)
        except Exception as e2:
            print(f"Error: Could not load Stable Diffusion model: {e2}")
            return False
    
    print("Model loaded successfully!")
    print()
    
    # Generate images
    generated_count = 0
    
    # Use progress bar
    with tqdm(total=NUM_IMAGES, desc="Generating images") as pbar:
        while generated_count < NUM_IMAGES:
            try:
                # Select a prompt (cycle through prompts)
                prompt = PROMPTS[generated_count % len(PROMPTS)]
                
                # Add some variation to the prompt
                if generated_count > 0:
                    prompt_variation = f"{prompt}, high quality, detailed"
                else:
                    prompt_variation = prompt
                
                # Generate image
                image = pipeline(
                    prompt=prompt_variation,
                    num_inference_steps=20,  # Reduced for faster generation
                    guidance_scale=7.5,
                    height=512,
                    width=512
                ).images[0]
                
                # Save image
                filename = f"diffusion_{generated_count:05d}.jpg"
                filepath = output_path / filename
                image.save(filepath, "JPEG", quality=95)
                
                generated_count += 1
                pbar.update(1)
                
                # Print progress every 50 images
                if generated_count % 50 == 0:
                    print(f"\nGenerated {generated_count}/{NUM_IMAGES} images...")
                    
            except Exception as e:
                print(f"\nError generating image {generated_count}: {e}")
                print("Continuing with next image...")
                continue
    
    print(f"\n✓ Successfully generated {generated_count} images!")
    print(f"✓ Images saved to: {output_path.absolute()}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Diffusion Image Generator")
    print("=" * 60)
    print()
    
    # Check if output directory exists
    if not os.path.exists("datasets"):
        print("Error: datasets directory not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    success = generate_images()
    
    if success:
        print("\n" + "=" * 60)
        print("Generation complete!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Generation failed. Please check the error messages above.")
        print("=" * 60)
        sys.exit(1)

