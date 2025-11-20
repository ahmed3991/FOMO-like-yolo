"""
Train on Full COCO Dataset

Script for training the FOMO model on the full COCO 2017 dataset.
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.train import train
from src.utils import Config


def main():
    parser = argparse.ArgumentParser(description='Train FOMO on Full COCO')
    
    # Model arguments
    parser.add_argument('--input-size', type=int, default=160, help='Input image size')
    parser.add_argument('--num-classes', type=int, default=80, help='Number of classes')
    
    # Dataset arguments
    parser.add_argument('--train-image-dir', type=str, default='data/coco/train/images',
                       help='Training images directory')
    parser.add_argument('--train-label-dir', type=str, default='data/coco/train/labels',
                       help='Training labels directory')
    parser.add_argument('--val-image-dir', type=str, default='data/coco/val/images',
                       help='Validation images directory')
    parser.add_argument('--val-label-dir', type=str, default='data/coco/val/labels',
                       help='Validation labels directory')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers')
    
    # Other arguments
    parser.add_argument('--quick-test', action='store_true', help='Quick test with small dataset')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--output-dir', type=str, default='checkpoints_coco',
                       help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Update config
    Config.TRAIN_IMAGE_DIR = args.train_image_dir
    Config.TRAIN_LABEL_DIR = args.train_label_dir
    Config.VAL_IMAGE_DIR = args.val_image_dir
    Config.VAL_LABEL_DIR = args.val_label_dir
    Config.NUM_CLASSES = args.num_classes
    Config.INPUT_SIZE = args.input_size
    Config.OUTPUT_SIZE = args.input_size // 8
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr
    Config.NUM_WORKERS = args.num_workers
    Config.CHECKPOINT_DIR = args.output_dir
    
    if args.quick_test:
        Config.QUICK_TEST = True
    
    # Train
    train(Config)


if __name__ == "__main__":
    main()
