# AI-Generated Image Detector - Training Steps

## ✅ Pre-Training Checklist

- [x] Dataset ready in `datasets/ai_detection/`
- [x] Real images: 31,783 images
- [x] GAN images: 70,001 images
- [x] Diffusion images: 2,778 images
- [ ] Notebook ready: `ai_generated_detection.ipynb`

## 📝 Step-by-Step Training Instructions

### Step 1: Open the Notebook
1. Open `ai_generated_detection.ipynb` in Jupyter Notebook/Lab
2. Make sure you're using the virtual environment (`venv`)

### Step 2: Run Imports (Cell 2)
- **What it does**: Imports all required libraries (TensorFlow, Keras, PIL, etc.)
- **Action**: Run the cell
- **Expected**: No errors, all imports successful
- **Time**: ~10-30 seconds

### Step 3: Set Configuration (Cell 4)
- **What it does**: Sets training parameters (image size, batch size, epochs, etc.)
- **Action**: Run the cell
- **Note**: You can adjust:
  - `BATCH_SIZE = 32` (reduce to 16 if out of memory)
  - `EPOCHS = 50` (can start with fewer for testing)
  - `LEARNING_RATE = 0.0001`
- **Expected**: Directories created, configuration set

### Step 4: Check for Existing Model (Cell 6) - **OPTIONAL**
- **What it does**: Checks if you already have a trained model
- **Action**: Run the cell
- **If model found**: You can skip training and go to evaluation
- **If no model**: Continue with training steps

### Step 5: Define Data Loading Functions (Cell 6)
- **What it does**: Defines functions to load images from folders
- **Action**: Run the cell
- **Expected**: Functions defined (no output)

### Step 6: Load Your Dataset (Cell 8) ⚠️ **IMPORTANT**
- **What it does**: Loads all images from your folder structure
- **Action**: Run the cell
- **Expected output**:
  ```
  Loading 31783 images from real...
  Loading 70001 images from gan...
  Loading 2778 images from diffusion...
  Dataset shape: (104562, 224, 224, 3)
  Labels shape: (104562,)
  
  Class distribution:
    Real: 31783 images
    GAN: 70001 images
    Diffusion: 2778 images
  ```
- **Time**: 5-15 minutes (depending on dataset size)
- **Note**: This might use a lot of RAM. If you run out of memory:
  - Reduce batch processing
  - Use fewer images per class

### Step 7: Data Preprocessing (Cell 10)
- **What it does**: 
  - Converts labels to categorical (one-hot encoding)
  - Splits data into train/validation (80/20)
- **Action**: Run the cell
- **Expected output**:
  ```
  Training samples: 83649
  Validation samples: 20913
  ```
- **Time**: ~1-2 minutes

### Step 8: Build Model Architecture (Cell 12)
- **What it does**: Creates EfficientNet-B4 model with custom classifier
- **Action**: Run the cell
- **Expected output**: Model summary showing architecture
- **Time**: ~1 minute (downloads EfficientNet weights on first run)

### Step 9: Setup Callbacks (Cell 14)
- **What it does**: Sets up:
  - ModelCheckpoint (saves best model during training)
  - EarlyStopping (stops if no improvement)
  - ReduceLROnPlateau (reduces learning rate)
- **Action**: Run the cell
- **Expected**: Callbacks created

### Step 10: Train the Model (Cell 16) 🚀 **LONG STEP**
- **What it does**: Trains the model on your dataset
- **Action**: Run the cell
- **Expected**: Training progress with epoch-by-epoch updates
- **Time**: 
  - **GPU**: 2-6 hours (depending on GPU)
  - **CPU**: 10-20+ hours (very slow, not recommended)
- **What to watch**:
  - Training accuracy should increase
  - Validation accuracy should increase
  - Loss should decrease
  - Model checkpoints will be saved automatically

**Training Output Example:**
```
Epoch 1/50
2614/2614 [==============================] - 120s 46ms/step - loss: 0.8234 - accuracy: 0.6234 - val_loss: 0.6543 - val_accuracy: 0.7123

Epoch 1: val_accuracy improved from -inf to 0.71234, saving model to models/ai_generated/ai_detector_best.keras
```

### Step 11: Save Final Model (Cell 18)
- **What it does**: Saves the final model state and training history
- **Action**: Run the cell
- **Expected output**:
  ```
  ✓ Final model saved to: models/ai_generated/ai_detector_final.keras
  ✓ Training history saved to: models/ai_generated/training_history.pkl
  
  Training Summary:
    Epochs completed: 25
    Final training accuracy: 0.9234
    Final validation accuracy: 0.8901
    Best validation accuracy: 0.8956
  ```
- **Time**: ~30 seconds

### Step 12: Evaluate Model (Cell 22) - **OPTIONAL**
- **What it does**: Evaluates model on validation set
- **Action**: Run the cell
- **Expected**: Classification report with precision, recall, F1-score

### Step 13: View Training Curves (Cell 26) - **OPTIONAL**
- **What it does**: Plots training/validation loss and accuracy curves
- **Action**: Run the cell
- **Expected**: Two plots showing training progress

### Step 14: View Confusion Matrix (Cell 28) - **OPTIONAL**
- **What it does**: Shows confusion matrix for all classes
- **Action**: Run the cell
- **Expected**: Visual confusion matrix

### Step 15: Test Single Image (Cell 24) - **OPTIONAL**
- **What it does**: Predicts a single image
- **Action**: Uncomment and modify the example code
- **Example**:
  ```python
  test_image_path = 'path/to/test/image.jpg'
  predicted_class, confidence, probs = predict_single_image(model, test_image_path)
  print(f"Predicted: {predicted_class}, Confidence: {confidence:.2%}")
  ```

---

## 🎯 Quick Start Commands

### If you want to run everything quickly:

```python
# In Jupyter Notebook, run cells in order:
# Cell 2 → Cell 4 → Cell 6 → Cell 8 → Cell 10 → Cell 12 → Cell 14 → Cell 16 → Cell 18
```

### Or use "Run All" (but monitor for errors):
- Menu: **Cell → Run All**
- Watch for any errors in data loading

---

## ⚙️ Configuration Adjustments

### If you run out of memory:
```python
# In Cell 4, change:
BATCH_SIZE = 16  # Reduce from 32
# Or reduce number of images loaded
```

### If training is too slow:
```python
# In Cell 4, change:
EPOCHS = 20  # Reduce from 50 for faster testing
```

### For balanced training:
Since you have imbalanced classes:
- Real: 31,783
- GAN: 70,001 (more)
- Diffusion: 2,778 (fewer)

The notebook uses stratified split which helps, but you might want to:
1. Use class weights (more advanced)
2. Or ensure you have roughly equal samples

---

## 📊 Expected Results

After training, you should see:

1. **Training Accuracy**: 85-95% (depending on dataset quality)
2. **Validation Accuracy**: 80-90%
3. **Model saved at**: `models/ai_generated/ai_detector_best.keras`

---

## 🐛 Troubleshooting

### Issue: "Out of Memory" error
**Solution**: 
- Reduce `BATCH_SIZE` to 16 or 8
- Close other applications
- Use fewer images

### Issue: "Folder does not exist" error
**Solution**: 
- Check folder structure: `datasets/ai_detection/real/`, `gan/`, `diffusion/`
- Ensure folder names are lowercase

### Issue: Training is very slow
**Solution**: 
- Use GPU if available
- Reduce number of images
- Reduce batch size
- Reduce epochs for testing

### Issue: Model accuracy is low
**Solution**: 
- Train for more epochs
- Check dataset quality
- Ensure balanced dataset
- Adjust learning rate

---

## ✅ Post-Training Checklist

After training completes:

- [ ] Model saved to `models/ai_generated/ai_detector_best.keras`
- [ ] Training history saved
- [ ] Validation accuracy > 80%
- [ ] Model can predict test images correctly

---

## 🚀 Next Steps After Training

1. **Test the model** on sample images using Cell 24
2. **Evaluate performance** using Cell 22 and 28
3. **Use in your main tool** by loading the saved model
4. **Integrate with other detectors** (ELA, PRNU, etc.)

---

## 📝 Notes

- **Training time**: Expect 2-6 hours on GPU, much longer on CPU
- **Model size**: ~200-300 MB
- **Best model**: Saved automatically during training based on validation accuracy
- **Resume training**: You can't directly resume, but you can load and fine-tune

Good luck with your training! 🎉

