# Tiny-VIN: FOMO-like YOLOv8 for Microcontrollers

A lightweight object detection model combining YOLOv8 architecture with FOMO-style heatmap outputs, optimized for deployment on microcontrollers like ESP32.

**Model Size**: ~590K parameters | **Input**: 160×160 or 96×96 | **Output**: Heatmap predictions

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Complete Training Workflow](#complete-training-workflow)
  - [Step 1: Pre-training on COCO128](#step-1-pre-training-on-coco128)
  - [Step 2: Transfer Learning on Grocery Dataset](#step-2-transfer-learning-on-grocery-dataset)
  - [Step 3: Evaluation & Visualization](#step-3-evaluation--visualization)
  - [Step 4: Model Export for Deployment](#step-4-model-export-for-deployment)
- [Quick Reference](#quick-reference)
- [Module Documentation](#module-documentation)
- [Configuration](#configuration)
- [License](#license)

---

## Features

✅ **Lightweight Architecture** - Only ~590K parameters, perfect for embedded devices  
✅ **FOMO-style Output** - Center-point heatmaps instead of bounding boxes  
✅ **Two-Stage Training** - Pre-train on COCO128, then transfer to custom dataset  
✅ **Multiple Export Formats** - ONNX, TFLite with INT8 quantization, C headers for ESP32  
✅ **Easy Workflow** - Simple commands via Makefile or direct script execution  
✅ **Professional Structure** - Clean, modular codebase following best practices  

---

## Project Structure

```
tiny-vin/
├── Makefile              # Workflow automation (make train, make export, etc.)
├── README.md             # This file
├── requirements.txt      # Python dependencies
│
├── src/                  # Source code (organized modules)
│   ├── datasets/         # Dataset handling + download utilities
│   ├── models/           # Model architecture + loss functions
│   ├── training/         # Training + transfer learning logic
│   ├── utils/            # Configuration management
│   ├── export/           # Model export (ONNX, TFLite, C headers)
│   └── evaluation/       # Metrics, verification, testing
│
├── scripts/              # CLI utilities (easy to run)
│   ├── train.py                    # Main training script
│   ├── transfer_learning.py       # Transfer learning script
│   ├── download_coco128.py         # Download COCO128 dataset
│   ├── download_kaggle_grocery.py  # Download Kaggle grocery dataset
│   ├── export_model.py             # Export model to all formats
│   ├── inference_example.py        # Quick inference demo
│   └── visualize_predictions.py    # Visualize model predictions
│
├── data/                 # Datasets (created after download)
├── checkpoints/          # Training checkpoints (best_model.pth)
├── checkpoints_transfer/ # Transfer learning checkpoints
└── models/              # Exported models (ONNX, TFLite, C headers)
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd tiny-vin
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Using Makefile
make install

# Or manually
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
make test-imports
```

You should see:
```
✓ Models import OK
✓ Datasets import OK
✓ Training import OK
✓ Export import OK
✓ Evaluation import OK
✓ All imports successful!
```

---

## Complete Training Workflow

This section explains the **recommended two-stage training approach**:
1. **Pre-train** on COCO128 (large, diverse dataset with 80 classes)
2. **Transfer learning** on your target dataset (e.g., grocery items with 43 classes)

This approach gives better results than training from scratch on a small dataset.

---

### Step 1: Pre-training on COCO128

**Goal**: Train a general-purpose object detector on COCO128 dataset.

#### 1.1 Download COCO128 Dataset

```bash
# Using Makefile (recommended)
make download-coco

# Or using script directly
python scripts/download_coco128.py --update-config --verify
```

This will:
- Download COCO128 dataset (~7.6 MB images)
- Extract to `data/coco128/`
- Create train/val split
- Update `src/utils/config.py` with dataset paths

**Dataset structure after download:**
```
data/coco128/
├── train/
│   ├── images/  # Training images
│   └── labels/  # YOLO format labels
└── val/
    ├── images/  # Validation images
    └── labels/  # YOLO format labels
```

#### 1.2 Train on COCO128

```bash
# Using Makefile (recommended)
make train

# Or using script directly
python scripts/train.py --epochs 30 --batch-size 16 --lr 0.001

# Quick test (2 epochs)
make train-quick
```

**Training parameters:**
- **Epochs**: 30 (recommended for good pre-training)
- **Batch size**: 16 (reduce if GPU memory limited)
- **Learning rate**: 0.001
- **Input size**: 160×160 pixels
- **Output**: 20×20 heatmap (160/8)
- **Classes**: 80 (COCO classes)

**What happens during training:**
1. Model trains on COCO128 images
2. Learns to detect 80 different object classes
3. Checkpoints saved to `checkpoints/`
4. Best model saved as `checkpoints/best_model.pth`
5. Training logs show loss, metrics every epoch

**Expected training time:**
- **CPU**: ~4-6 hours
- **GPU (CUDA)**: ~30-45 minutes
- **Mac (MPS)**: ~1-2 hours

**Monitor training:**
```
Epoch 1/30: 100%|████████| 25/25 [00:45<00:00]  
  Train Loss: 0.6234 | Val Loss: 0.5891
  Precision: 0.723 | Recall: 0.681 | F1: 0.701
Epoch 2/30: 100%|████████| 25/25 [00:43<00:00]
  Train Loss: 0.5123 | Val Loss: 0.4956
  ...
```

#### 1.3 Verify Pre-trained Model

```bash
# Check model architecture
make verify

# Visualize predictions on COCO128 validation set
make visualize

# Or with custom parameters
make visualize CHECKPOINT=checkpoints/best_model.pth \
               IMAGES=data/coco128/val/images \
               NUM_SAMPLES=10
```

**Output**: Predictions saved to `predictions/` directory with visualization grid.

---

### Step 2: Transfer Learning on Grocery Dataset

**Goal**: Fine-tune the pre-trained model on your specific dataset (grocery items).

#### 2.1 Download Grocery Dataset

```bash
# Using Makefile
make download-kaggle

# Or using script directly
python scripts/download_kaggle_grocery.py --download --prepare
```

**Note**: You need Kaggle credentials. The script will guide you through setup:
1. Create account at [kaggle.com](https://www.kaggle.com)
2. Go to Account Settings → API → Create New API Token
3. Place `kaggle.json` in `~/.kaggle/`

**Dataset structure after download:**
```
data/grocery_items/
├── train/
│   ├── images/  # Training images
│   └── labels/  # YOLO format labels
├── valid/
│   ├── images/  # Validation images
│   └── labels/  # YOLO format labels
└── num_classes.txt  # Number of classes (e.g., 43)
```

#### 2.2 Run Transfer Learning

```bash
# Using Makefile (simplest)
make transfer CHECKPOINT=checkpoints/best_model.pth \
              DATA=data/grocery_items \
              CLASSES=43 \
              EPOCHS=20

# Or using script directly
python scripts/transfer_learning.py \
    --checkpoint checkpoints/best_model.pth \
    --train-image-dir data/grocery_items/train/images \
    --train-label-dir data/grocery_items/train/labels \
    --val-image-dir data/grocery_items/valid/images \
    --val-label-dir data/grocery_items/valid/labels \
    --num-classes 43 \
    --epochs 20 \
    --lr 0.0001 \
    --batch-size 16
```

**Key parameters:**
- **Checkpoint**: `checkpoints/best_model.pth` (pre-trained on COCO128)
- **Epochs**: 20 (less than pre-training since we're fine-tuning)
- **Learning rate**: 0.0001 (10x smaller than pre-training)
- **Classes**: 43 (grocery dataset classes)
- **Batch size**: 16

**What happens during transfer learning:**
1. Loads pre-trained model from COCO128
2. Replaces final layer for 43 classes (was 80)
3. Fine-tunes all layers with lower learning rate
4. Saves checkpoints to `checkpoints_transfer/`
5. Best model: `checkpoints_transfer/best_model_transfer.pth`

**Expected training time:**
- **CPU**: ~2-3 hours
- **GPU (CUDA)**: ~15-20 minutes
- **Mac (MPS)**: ~30-45 minutes

**Why transfer learning works:**
- Pre-trained model already knows how to detect objects
- Only needs to adapt to grocery-specific features
- Converges faster with less data
- Better accuracy than training from scratch

---

### Step 3: Evaluation & Visualization

#### 3.1 Visualize Predictions

```bash
# Visualize transfer learning results
make visualize CHECKPOINT=checkpoints_transfer/best_model_transfer.pth \
               IMAGES=data/grocery_items/valid/images \
               NUM_SAMPLES=20

# Or using script
python scripts/visualize_predictions.py \
    --checkpoint checkpoints_transfer/best_model_transfer.pth \
    --image-dir data/grocery_items/valid/images \
    --label-dir data/grocery_items/valid/labels \
    --num-samples 20 \
    --output-dir predictions_grocery \
    --save-individual \
    --create-grid
```

**Output:**
- Individual prediction images in `predictions_grocery/`
- Combined grid: `predictions_grocery/predictions_grid.png`

#### 3.2 Run Inference Demo

```bash
# Quick demo with checkpoint
python scripts/inference_example.py

# Or with specific image
python scripts/inference_example.py \
    --image data/grocery_items/valid/images/sample.jpg \
    --label data/grocery_items/valid/labels/sample.txt
```

---

### Step 4: Model Export for Deployment

Export your trained model for deployment on ESP32 or other embedded devices.

#### 4.1 Export to All Formats

```bash
# Export to ONNX, TFLite (quantized), and C header
make export CHECKPOINT=checkpoints_transfer/best_model_transfer.pth

# Or using script
python scripts/export_model.py \
    --checkpoint checkpoints_transfer/best_model_transfer.pth \
    --all \
    --output-dir models
```

This creates:
- `models/model.onnx` - ONNX format
- `models/model_quantized.tflite` - TFLite with INT8 quantization
- `models/model.h` - C header file for ESP32

#### 4.2 Export Specific Formats

```bash
# ONNX only
make export-onnx CHECKPOINT=checkpoints_transfer/best_model_transfer.pth

# TFLite with quantization only
make export-tflite CHECKPOINT=checkpoints_transfer/best_model_transfer.pth
```

#### 4.3 Verify Exported Model

The export script automatically verifies the TFLite model:
```
Model details:
  Input shape:  [1, 160, 160, 3]
  Input dtype:  <class 'numpy.uint8'>
  Output shape: [1, 20, 20, 43]
  Output dtype: <class 'numpy.uint8'>
✓ Inference successful!
```

**Model sizes:**
- ONNX: ~2.3 MB
- TFLite (quantized): ~590 KB (**fits ESP32!**)
- C header: ~590 KB

---

## Quick Reference

### Common Commands

```bash
# Setup
make install              # Install dependencies
make test-imports         # Verify installation

# Pre-training on COCO128
make download-coco        # Download dataset
make train                # Train (30 epochs)
make train-quick          # Quick test (2 epochs)

# Transfer learning on grocery dataset
make download-kaggle      # Download dataset
make transfer             # Fine-tune model

# Visualization & evaluation
make visualize            # Generate predictions
make demo                 # Run inference demo

# Export for deployment
make export               # Export all formats
make export-onnx          # ONNX only
make export-tflite        # TFLite only

# Maintenance
make clean                # Clean cache
make clean-all            # Clean everything
make help                 # Show all commands
```

### Direct Script Execution

All scripts can be run directly without Makefile:

```bash
python scripts/train.py --epochs 30
python scripts/transfer_learning.py --checkpoint checkpoints/best_model.pth --num-classes 43
python scripts/visualize_predictions.py --checkpoint checkpoints/best_model.pth
python scripts/export_model.py --checkpoint checkpoints/best_model.pth --all
```

---

## Module Documentation

### `src/datasets/`
- **dataset.py**: `FOMODataset`, `SimpleAugmentDataset` classes
- **data_utils.py**: YOLO label loading, heatmap conversion utilities
- **download/**: Dataset download utilities (COCO, Kaggle, Roboflow)

### `src/models/`
- **architecture.py**: `MicroYOLO` model definition (~590K parameters)
- **loss.py**: `FOMOLoss`, `FocalFOMOLoss`, `WeightedFOMOLoss`

### `src/training/`
- **train.py**: Main training loop with data loading, optimization
- **transfer_learning.py**: Transfer learning with layer freezing/unfreezing

### `src/export/`
- **onnx.py**: PyTorch → ONNX export
- **tflite.py**: ONNX → TensorFlow → TFLite conversion with quantization
- **c_header.py**: Binary model → C header for embedded deployment

### `src/evaluation/`
- **metrics.py**: Centroid extraction, precision/recall/F1 metrics
- **verify.py**: Model architecture verification
- **test_tflite.py**: TFLite model testing utilities

---

## Configuration

Configuration is managed in `src/utils/config.py`. Key parameters:

```python
class Config:
    # Model
    INPUT_SIZE = 160          # Input image size (160 or 96)
    NUM_CLASSES = 2           # Number of object classes
    
    # Training
    BATCH_SIZE = 16           # Batch size
    NUM_EPOCHS = 30           # Training epochs
    LEARNING_RATE = 0.001     # Initial learning rate
    
    # Loss function
    LOSS_TYPE = "bce"         # "bce", "focal", or "weighted"
    
    # Device
    DEVICE = "cuda" if available else "mps" if available else "cpu"
```

Override via command-line arguments:
```bash
python scripts/train.py --batch-size 32 --lr 0.0005 --epochs 50
```

---

## Training Tips

### For Best Results:

1. **Always pre-train on COCO128 first**
   - Provides strong feature extraction
   - Better generalization
   - Faster convergence on target dataset

2. **Use appropriate learning rates**
   - Pre-training: 0.001
   - Transfer learning: 0.0001 (10x smaller)

3. **Adjust batch size for your hardware**
   - GPU with 8GB+ VRAM: Use 16-32
   - GPU with 4-6GB VRAM: Use 8-16
   - CPU: Use 4-8

4. **Monitor validation loss**
   - Should decrease over time
   - If increasing: reduce learning rate
   - If plateauing: may need more epochs

5. **Data augmentation**
   - FOMODataset includes random flips, crops
   - Helps prevent overfitting
   - Important for small datasets

---

## Troubleshooting

### Model not improving?
- Reduce learning rate by 10x
- Increase number of epochs
- Check data quality (verify annotations)

### Out of memory?
- Reduce batch size
- Use smaller input size (96×96 instead of 160×160)
- Use CPU instead of GPU

### Dataset download fails?
- Check internet connection
- For Kaggle: Verify credentials in `~/.kaggle/kaggle.json`
- For Roboflow: Check API key

### Export fails?
- Ensure checkpoint exists
- Check TensorFlow installation: `pip install tensorflow`
- Check ONNX installation: `pip install onnx onnx-tf`

---

## Citation

If you use this code, please cite:

```
@misc{tiny-vin-fomo-yolo,
  author = {Your Name},
  title = {Tiny-VIN: FOMO-like YOLOv8 for Microcontrollers},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/tiny-vin}
}
```

---

## License

[Your License Here - e.g., MIT, Apache 2.0]

---

## Acknowledgments

- YOLOv8 architecture inspiration
- FOMO (Faster Objects, More Objects) approach
- COCO dataset
- Kaggle community for datasets

---

**Happy Training! 🚀**

For questions or issues, please open an issue on GitHub.
