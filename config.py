"""
Project Configuration

This file contains path configurations for datasets and training.
It is automatically updated by download scripts.
"""

# Dataset Paths
# These are updated automatically when running download scripts
TRAIN_IMAGE_DIR = "data/coco128/train/images"
TRAIN_LABEL_DIR = "data/coco128/train/labels"
VAL_IMAGE_DIR = "data/coco128/val/images"
VAL_LABEL_DIR = "data/coco128/val/labels"

# Model Configuration
INPUT_SIZE = 160  # 160x160 or 96x96
NUM_CLASSES = 80  # COCO default, updated based on dataset
