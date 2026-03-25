# Step-by-Step Training Guide

## Quick Start: Essential Cells to Run

### **STEP 1: Setup & Imports** (Run Once)
- **Cell 4**: Import all required libraries
  - This loads pandas, numpy, matplotlib, keras, tensorflow, etc.
  - ⚠️ **MUST RUN FIRST**

- **Cell 6**: Import PIL (Python Imaging Library)
  - Required for image processing

### **STEP 2: Define Functions** (Run Once)
- **Cell 8**: `get_imlist()` function (optional, not used in current workflow)
- **Cell 9**: `convert_to_ela_image()` function
  - ⚠️ **REQUIRED** - This converts images to ELA format
  - This is the core preprocessing function

### **STEP 3: Data Loading & Preprocessing** (Run Once)
- **Cell 23**: Load the dataset CSV
  ```python
  dataset = pd.read_csv('datasets/dataset.csv')
  ```

- **Cell 24**: Initialize empty lists
  ```python
  X = []
  Y = []
  ```

- **Cell 25**: ⚠️ **IMPORTANT - This takes time!**
  - Converts all images to ELA format
  - Processes ~9,500 images
  - **Expected time: 10-30 minutes depending on your system**
  ```python
  for index, row in dataset.iterrows():
      X.append(array(convert_to_ela_image(row[0], 90).resize((128, 128))).flatten() / 255.0)
      Y.append(row[1])
  ```

- **Cell 27**: Normalize and convert labels
  ```python
  X = np.array(X)
  Y = to_categorical(Y, 2)
  ```

- **Cell 29**: Reshape data for CNN
  ```python
  X = X.reshape(-1, 128, 128, 3)
  ```

### **STEP 4: Train-Test Split**
- **Cell 31**: Split data into training and validation sets
  ```python
  X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)
  ```

### **STEP 5: Build the Model**
- **Cell 33**: Define the CNN architecture
  - Creates the convolutional neural network
  - Shows input/output shapes

- **Cell 34**: View model summary (optional but recommended)
  - Shows total parameters and architecture details

- **Cell 36**: Define optimizer
  ```python
  optimizer = RMSprop(learning_rate=0.0005, rho=0.9, epsilon=1e-08)
  ```

- **Cell 37**: Compile the model
  ```python
  model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
  ```

- **Cell 39**: Define early stopping callback
  ```python
  early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, verbose=0, mode='auto')
  ```

### **STEP 6: Train the Model** ⚠️ **MAIN TRAINING STEP**
- **Cell 41**: Set training parameters (optional, can modify)
  ```python
  epochs = 30
  batch_size = 100
  ```

- **Cell 42**: ⚠️ **START TRAINING HERE**
  - This is where the actual training happens
  - **Expected time: 30-60 minutes** (depends on your GPU/CPU)
  - Training will stop early if validation accuracy doesn't improve for 2 epochs
  ```python
  history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
            validation_data = (X_val, Y_val), verbose = 2, callbacks=[early_stopping])
  ```

### **STEP 7: Evaluate Results** (Optional but Recommended)
- **Cell 45**: Plot training curves (loss and accuracy)
- **Cell 47**: Generate confusion matrix

---

## Summary: Minimum Cells to Run

**In order:**
1. Cell 4 (imports)
2. Cell 6 (PIL imports)
3. Cell 9 (ELA function)
4. Cell 23 (load CSV)
5. Cell 24 (initialize lists)
6. Cell 25 (process images - **takes time!**)
7. Cell 27 (normalize)
8. Cell 29 (reshape)
9. Cell 31 (train-test split)
10. Cell 33 (build model)
11. Cell 36 (optimizer)
12. Cell 37 (compile)
13. Cell 39 (early stopping)
14. Cell 41 (set parameters)
15. **Cell 42 (TRAIN - main step!)**

---

## Optional Cells (Can Skip)

- Cells 0-3: Documentation/headers
- Cells 10-20: Sample image visualization (for understanding, not required)
- Cell 34: Model summary (informational)
- Cells 45, 47: Visualization (run after training to see results)

---

## Tips

1. **Run cells sequentially** - Don't skip cells as variables depend on previous cells
2. **Cell 25 is slow** - Be patient, it processes all images
3. **Cell 42 is the training** - This is where the model learns
4. **Monitor the output** - Watch for validation accuracy improvements
5. **Early stopping** - Training stops automatically if no improvement for 2 epochs

---

## Expected Output

After running Cell 42, you should see:
```
Train on ~7600 samples, validate on ~1900 samples
Epoch 1/30
 - Xs - loss: X.XXXX - accuracy: X.XXXX - val_loss: X.XXXX - val_accuracy: X.XXXX
...
```

The model will train until:
- All 30 epochs complete, OR
- Early stopping triggers (no improvement for 2 epochs)

