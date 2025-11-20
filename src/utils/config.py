"""
Configuration file for FOMO-like YOLOv8 training.

Centralized configuration for model, training, and dataset parameters.
"""

import torch
from pathlib import Path


class Config:
    """Configuration class for FOMO training."""
    
    # ========== Model Configuration ==========
    INPUT_SIZE = 160  # Input image size (160 or 96)
    NUM_CLASSES = 2   # Number of object classes
    OUTPUT_SIZE = INPUT_SIZE // 8  # Heatmap size (20 for 160, 12 for 96)
    
    # ========== Dataset Configuration ==========
    # TODO: Update these paths to your dataset
    TRAIN_IMAGE_DIR = "data/coco128/train/images"
    TRAIN_LABEL_DIR = "data/coco128/train/labels"
    VAL_IMAGE_DIR = "data/coco128/val/images"
    VAL_LABEL_DIR = "data/coco128/val/labels"
    
    # Data augmentation
    USE_AUGMENTATION = True
    USE_ALBUMENTATIONS = False  # Set to True if albumentations is installed
    
    # ========== Training Configuration ==========
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "cosine"  # "cosine" or "step"
    MIN_LR = 1e-6  # For cosine annealing
    
    # Loss function
    LOSS_TYPE = "bce"  # "bce", "focal", or "weighted"
    FOCAL_ALPHA = 0.25  # For focal loss
    FOCAL_GAMMA = 2.0   # For focal loss
    POS_WEIGHT = 10.0   # Positive sample weight for BCE (or None)
    
    # ========== Optimizer Configuration ==========
    OPTIMIZER = "adamw"  # "adam" or "adamw"
    
    # ========== Device Configuration ==========
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    NUM_WORKERS = 4  # DataLoader workers
    PIN_MEMORY = True if DEVICE == "cuda" else False
    
    # ========== Checkpoint Configuration ==========
    CHECKPOINT_DIR = "checkpoints"
    SAVE_FREQUENCY = 5  # Save checkpoint every N epochs
    SAVE_BEST_ONLY = True  # Only save if validation loss improves
    
    # ========== Logging Configuration ==========
    LOG_DIR = "logs"
    LOG_FREQUENCY = 10  # Log every N batches
    USE_TENSORBOARD = False
    
    # ========== Validation Configuration ==========
    VAL_FREQUENCY = 1  # Validate every N epochs
    
    # ========== Quick Test Mode ==========
    QUICK_TEST = False  # Use small subset for quick testing
    QUICK_TEST_SAMPLES = 20
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("=" * 70)
        print("FOMO Training Configuration")
        print("=" * 70)
        
        print("\n[Model]")
        print(f"  Input size:      {cls.INPUT_SIZE}x{cls.INPUT_SIZE}")
        print(f"  Output size:     {cls.OUTPUT_SIZE}x{cls.OUTPUT_SIZE}")
        print(f"  Number of classes: {cls.NUM_CLASSES}")
        
        print("\n[Dataset]")
        print(f"  Train images:    {cls.TRAIN_IMAGE_DIR}")
        print(f"  Train labels:    {cls.TRAIN_LABEL_DIR}")
        print(f"  Val images:      {cls.VAL_IMAGE_DIR}")
        print(f"  Val labels:      {cls.VAL_LABEL_DIR}")
        print(f"  Augmentation:    {cls.USE_AUGMENTATION}")
        print(f"  Albumentations:  {cls.USE_ALBUMENTATIONS}")
        
        print("\n[Training]")
        print(f"  Batch size:      {cls.BATCH_SIZE}")
        print(f"  Epochs:          {cls.NUM_EPOCHS}")
        print(f"  Learning rate:   {cls.LEARNING_RATE}")
        print(f"  Weight decay:    {cls.WEIGHT_DECAY}")
        print(f"  Optimizer:       {cls.OPTIMIZER}")
        print(f"  Loss type:       {cls.LOSS_TYPE}")
        print(f"  Scheduler:       {cls.SCHEDULER_TYPE if cls.USE_SCHEDULER else 'None'}")
        
        print("\n[Device]")
        print(f"  Device:          {cls.DEVICE}")
        print(f"  Num workers:     {cls.NUM_WORKERS}")
        
        print("\n[Checkpointing]")
        print(f"  Checkpoint dir:  {cls.CHECKPOINT_DIR}")
        print(f"  Save frequency:  {cls.SAVE_FREQUENCY} epochs")
        print(f"  Save best only:  {cls.SAVE_BEST_ONLY}")
        
        print("\n" + "=" * 70)
    
    @classmethod
    def from_args(cls, args):
        """Update configuration from command-line arguments."""
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            cls.BATCH_SIZE = args.batch_size
        if hasattr(args, 'epochs') and args.epochs is not None:
            cls.NUM_EPOCHS = args.epochs
        if hasattr(args, 'lr') and args.lr is not None:
            cls.LEARNING_RATE = args.lr
        if hasattr(args, 'input_size') and args.input_size is not None:
            cls.INPUT_SIZE = args.input_size
            cls.OUTPUT_SIZE = args.input_size // 8
        if hasattr(args, 'num_classes') and args.num_classes is not None:
            cls.NUM_CLASSES = args.num_classes
        if hasattr(args, 'quick_test') and args.quick_test:
            cls.QUICK_TEST = True
        
        # Dataset paths
        if hasattr(args, 'train_image_dir') and args.train_image_dir is not None:
            cls.TRAIN_IMAGE_DIR = args.train_image_dir
        if hasattr(args, 'train_label_dir') and args.train_label_dir is not None:
            cls.TRAIN_LABEL_DIR = args.train_label_dir
        if hasattr(args, 'val_image_dir') and args.val_image_dir is not None:
            cls.VAL_IMAGE_DIR = args.val_image_dir
        if hasattr(args, 'val_label_dir') and args.val_label_dir is not None:
            cls.VAL_LABEL_DIR = args.val_label_dir


if __name__ == "__main__":
    # Print default configuration
    Config.print_config()
