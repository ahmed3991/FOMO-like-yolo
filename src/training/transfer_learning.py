"""
Transfer Learning Script for FOMO Model

This script loads a pre-trained checkpoint and fine-tunes on a new dataset.

Usage:
    python transfer_learning.py --checkpoint checkpoints/best_model.pth --epochs 20
"""

import os
import argparse
from pathlib import Path

import torch

from src.models import MicroYOLO
from src.datasets import FOMODataset, SimpleAugmentDataset
from src.utils import Config


def load_pretrained_model(checkpoint_path, num_classes_new, input_size=160, device='cpu'):
    """
    Load pre-trained model and adapt for new number of classes.
    
    Args:
        checkpoint_path: Path to pre-trained checkpoint
        num_classes_new: Number of classes in new dataset
        input_size: Input image size
        device: Device to load model on
        
    Returns:
        model: Model with transferred weights
        epoch_start: Epoch to start from (for continued training)
    """
    print("=" * 70)
    print("Loading Pre-trained Model for Transfer Learning")
    print("=" * 70)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes_old = checkpoint['config']['num_classes']
    
    print(f"\nPre-trained model info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    print(f"  Old classes: {num_classes_old}")
    print(f"  New classes: {num_classes_new}")
    
    # Create model with OLD number of classes first
    model_old = MicroYOLO(nc=num_classes_old, input_size=input_size)
    model_old.load_state_dict(checkpoint['model_state_dict'])
    
    # Create new model with NEW number of classes
    model_new = MicroYOLO(nc=num_classes_new, input_size=input_size)
    
    # Transfer weights
    print("\nTransferring weights...")
    
    # Copy all weights except the final classification layer
    model_new_dict = model_new.state_dict()
    pretrained_dict = {k: v for k, v in model_old.state_dict().items() 
                      if k in model_new_dict and 'head_conv2' not in k}
    
    model_new_dict.update(pretrained_dict)
    model_new.load_state_dict(model_new_dict)
    
    print(f"✓ Transferred {len(pretrained_dict)} layers")
    print(f"✗ Skipped final layer (head_conv2) - will be trained from scratch")
    
    # Optionally freeze backbone for initial training
    # Uncomment to freeze backbone:
    # for name, param in model_new.named_parameters():
    #     if 'head' not in name:  # Freeze everything except head
    #         param.requires_grad = False
    
    return model_new, 0


def transfer_learn():
    """Main transfer learning function."""
    parser = argparse.ArgumentParser(description='Transfer learning for FOMO model')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to pre-trained checkpoint')
    
    # Dataset arguments
    parser.add_argument('--train-image-dir', type=str, required=True,
                       help='Training images directory')
    parser.add_argument('--train-label-dir', type=str, required=True,
                       help='Training labels directory')
    parser.add_argument('--val-image-dir', type=str, required=True,
                       help='Validation images directory')
    parser.add_argument('--val-label-dir', type=str, required=True,
                       help='Validation labels directory')
    
    # Training arguments
    parser.add_argument('--num-classes', type=int, required=True,
                       help='Number of classes in new dataset')
    parser.add_argument('--input-size', type=int, default=160,
                       help='Input size')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (use lower for transfer learning)')
    
    # Other arguments
    parser.add_argument('--output-dir', type=str, default='checkpoints_transfer',
                       help='Output directory for checkpoints')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with small subset')
    
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
    Config.CHECKPOINT_DIR = args.output_dir
    
    if args.quick_test:
        Config.QUICK_TEST = True
    
    # Print configuration
    Config.print_config()
    
    # Set device
    device = torch.device(Config.DEVICE)
    print(f"\nUsing device: {device}")
    
    # Import training function
    from .train import train_one_epoch, validate, save_checkpoint, create_dataloaders, create_loss_function, create_optimizer, create_scheduler
    import time
    from tqdm import tqdm
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(Config)
    
    # Load model
    model, start_epoch = load_pretrained_model(
        args.checkpoint, 
        args.num_classes, 
        args.input_size, 
        device
    )
    model.to(device)
    
    # Create loss, optimizer, scheduler
    criterion = create_loss_function(Config)
    criterion = criterion.to(device)
    optimizer = create_optimizer(model, Config)
    scheduler = create_scheduler(optimizer, Config, Config.NUM_EPOCHS)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, Config
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch, Config)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = Config.LEARNING_RATE
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✓ New best validation loss!")
            save_checkpoint(model, optimizer, epoch, val_loss, Config, "best_model_transfer.pth")
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Transfer Learning Complete!")
    print("=" * 70)
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    save_checkpoint(model, optimizer, Config.NUM_EPOCHS - 1, val_loss, Config, "final_model_transfer.pth")
    
    print(f"\nCheckpoints saved to: {Config.CHECKPOINT_DIR}/")
    print("  - best_model_transfer.pth")
    print("  - final_model_transfer.pth")


if __name__ == "__main__":
    transfer_learn()
