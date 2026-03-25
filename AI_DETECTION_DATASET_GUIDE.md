# AI-Generated Image Detection - Dataset Guide

## 📁 Required Folder Structure

Your dataset **MUST** be organized in the following structure:

```
datasets/
└── ai_detection/
    ├── real/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── gan/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── diffusion/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## 📊 Dataset Requirements

### Minimum Requirements (Recommended for Testing):
- **Real images**: 1,000+ images
- **GAN-generated**: 1,000+ images  
- **Diffusion-generated**: 1,000+ images

### Ideal Requirements (For Better Accuracy):
- **Real images**: 5,000-10,000+ images
- **GAN-generated**: 5,000-10,000+ images
- **Diffusion-generated**: 5,000-10,000+ images

## 🎯 Recommended Datasets

### 1. Real Images (Natural Photos)

#### Option A: COCO Dataset (Recommended)
- **Link**: https://cocodataset.org/#download
- **Download**: 
  - Go to "2017 Train/Val images" section
  - Download "train2017.zip" (18GB) or "val2017.zip" (1GB)
- **Format**: Extract zip, images are in folders
- **Usage**: Use subset of images (5000-10000 images)
- **Quality**: High quality, diverse real-world images

#### Option B: ImageNet Subset
- **Link**: https://www.image-net.org/download.php
- **Note**: Requires registration, large download
- **Usage**: Download subset (about 10,000 images)

#### Option C: Flickr30k/Flickr8k
- **Link**: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
- **Format**: Images in folders
- **Usage**: Easy to download from Kaggle

### 2. GAN-Generated Images

#### Option A: CelebA-DF (Face-focused)
- **Link**: https://github.com/yuezunli/celeb-deepfakeforensics
- **Format**: Images with labels
- **Usage**: Extract fake images from dataset
- **Note**: Good for face-specific detection

#### Option B: ProGAN/StyleGAN Generated Images
- **Link**: https://github.com/NVlabs/stylegan
- **How to get**: 
  1. Download StyleGAN2 model
  2. Generate 5000+ fake images using the model
  3. Save to `gan/` folder
- **Script**: Use StyleGAN's generate.py to create images

#### Option C: FaceForensics++ (For Faces)
- **Link**: https://github.com/ondyari/FaceForensics
- **Format**: Video frames (extract frames)
- **Usage**: Extract fake frames from videos

#### Option D: AI-Generated Images Dataset (Kaggle)
- **Link**: Search "AI generated images" on Kaggle
- **Several datasets available**
- **Format**: Usually organized or easy to organize

### 3. Diffusion Model Images (DALL-E, Midjourney, Stable Diffusion)

#### Option A: Create Your Own Collection (Recommended)
1. **Stable Diffusion**:
   - Use Hugging Face models
   - Generate 3000-5000 images
   - Script example:
   ```python
   from diffusers import StableDiffusionPipeline
   import torch
   
   pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
   
   for i in range(5000):
       image = pipeline("a beautiful landscape").images[0]
       image.save(f"datasets/ai_detection/diffusion/image_{i}.jpg")
   ```

2. **DALL-E Images**:
   - Use DALL-E API to generate images
   - Or download from public galleries
   - Save to `diffusion/` folder

3. **Midjourney Images**:
   - Download from Midjourney gallery
   - Public collections available online

#### Option B: DiffusionDB (Large Collection)
- **Link**: https://poloclub.github.io/diffusiondb/
- **Note**: Large dataset, may need subset
- **Format**: Download and extract images

#### Option C: LAION Dataset (May contain AI images)
- **Link**: https://laion.ai/blog/laion-5b/
- **Note**: Very large, filter for AI-generated

## 📥 Step-by-Step Dataset Setup

### Method 1: Quick Start (Using Public Datasets)

1. **Create folder structure**:
   ```bash
   mkdir -p datasets/ai_detection/real
   mkdir -p datasets/ai_detection/gan
   mkdir -p datasets/ai_detection/diffusion
   ```

2. **Download Real Images**:
   ```bash
   # Download COCO val2017 (smaller, faster)
   wget http://images.cocodataset.org/zips/val2017.zip
   unzip val2017.zip
   # Copy 5000 random images to real folder
   shuf -zn1000 -e val2017/*.jpg | xargs -0 cp -t datasets/ai_detection/real/
   ```

3. **Download/Generate GAN Images**:
   - Use StyleGAN to generate images OR
   - Download from CelebA-DF dataset

4. **Generate Diffusion Images**:
   - Use Stable Diffusion (see script above) OR
   - Download from DiffusionDB

### Method 2: Using Kaggle Datasets

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   kaggle datasets download -d DATASET_NAME
   ```

2. **Search for datasets**:
   - "AI generated images"
   - "GAN generated images"
   - "deepfake dataset"
   - "synthetic images"

3. **Organize downloaded images**:
   - Move real images to `real/`
   - Move GAN to `gan/`
   - Move diffusion to `diffusion/`

### Method 3: Create Balanced Custom Dataset

Use this Python script to organize your images:

```python
import os
import shutil
import random
from pathlib import Path

def organize_dataset(source_dir, target_dir, class_name, num_images=5000):
    """
    Copy random images from source to target class folder
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir) / class_name
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(source_path.glob('*.jpg')) + \
                  list(source_path.glob('*.png'))
    
    # Randomly sample
    selected = random.sample(image_files, min(num_images, len(image_files)))
    
    # Copy to target
    for img in selected:
        shutil.copy(img, target_path / img.name)
    
    print(f"Copied {len(selected)} images to {target_path}")

# Usage
organize_dataset('coco_images', 'datasets/ai_detection', 'real', 5000)
organize_dataset('gan_images', 'datasets/ai_detection', 'gan', 5000)
organize_dataset('diffusion_images', 'datasets/ai_detection', 'diffusion', 5000)
```

## 🔍 Verify Dataset Structure

Run this to verify your dataset:

```python
import os

dataset_dir = 'datasets/ai_detection'
classes = ['real', 'gan', 'diffusion']

for class_name in classes:
    class_path = os.path.join(dataset_dir, class_name)
    if os.path.exists(class_path):
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"{class_name}: {len(images)} images")
    else:
        print(f"{class_name}: FOLDER NOT FOUND!")
```

## 📝 Important Notes

1. **Image Formats**: Supports `.jpg`, `.jpeg`, `.png`, `.bmp`
2. **Image Size**: Will be automatically resized to 224x224 during loading
3. **Balance**: Try to have roughly equal number of images in each class
4. **Quality**: Use high-quality images for better training
5. **Diversity**: Include diverse images (different scenes, objects, styles)

## 🚀 Quick Start with Minimum Dataset

If you want to test quickly with minimal data:

1. **Real images**: 100 images from COCO
2. **GAN images**: Generate 100 using StyleGAN
3. **Diffusion images**: Generate 100 using Stable Diffusion

Then train with smaller batch size and fewer epochs.

## 📚 Additional Resources

- **StyleGAN2**: https://github.com/NVlabs/stylegan2
- **Stable Diffusion**: https://huggingface.co/docs/diffusers
- **COCO Dataset**: https://cocodataset.org/
- **FaceForensics++**: https://github.com/ondyari/FaceForensics

## ⚠️ Common Issues

1. **"Folder does not exist" error**:
   - Make sure folder structure is exactly: `datasets/ai_detection/real/`, etc.
   - Check folder names are lowercase: `real`, `gan`, `diffusion`

2. **No images found**:
   - Check file extensions (.jpg, .png, etc.)
   - Make sure images are actually in the folders

3. **Out of memory**:
   - Reduce batch size in notebook (change BATCH_SIZE)
   - Use fewer images
   - Process images in smaller batches

## ✅ Checklist Before Training

- [ ] Created folder structure: `datasets/ai_detection/real/`, `gan/`, `diffusion/`
- [ ] At least 1000 images per class (more is better)
- [ ] Verified images are in correct folders
- [ ] Images are in supported formats (.jpg, .png, etc.)
- [ ] Have enough disk space (models can be large)

