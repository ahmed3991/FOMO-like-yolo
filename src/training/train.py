"""
Training script for FOMO-like YOLOv8 Model.

Main training loop with checkpointing, validation, and logging.
"""

import os
import argparse
from pathlib import Path
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from src.models import MicroYOLO
from src.datasets import FOMODataset, SimpleAugmentDataset
from src.models import FOMOLoss, FocalFOMOLoss, WeightedFOMOLoss
from src.utils import Config


def create_dataloaders(config):
    """Create train and validation dataloaders."""
    
    # Choose dataset class based on config and availability
    if config.USE_ALBUMENTATIONS:
        try:
            from src.datasets import FOMODataset
            DatasetClass = FOMODataset
            print("Using FOMODataset with albumentations")
        except ImportError:
            from src.datasets import SimpleAugmentDataset
            DatasetClass = SimpleAugmentDataset
            print("⚠️  Albumentations not available, using SimpleAugmentDataset")
    else:
        from src.datasets import SimpleAugmentDataset
        DatasetClass = SimpleAugmentDataset
        print("Using SimpleAugmentDataset (basic augmentations)")
    
    # Create datasets
    train_dataset = DatasetClass(
        image_dir=config.TRAIN_IMAGE_DIR,
        label_dir=config.TRAIN_LABEL_DIR,
        input_size=config.INPUT_SIZE,
        num_classes=config.NUM_CLASSES,
        augment=config.USE_AUGMENTATION,
        normalize=True
    )
    
    val_dataset = DatasetClass(
        image_dir=config.VAL_IMAGE_DIR,
        label_dir=config.VAL_LABEL_DIR,
        input_size=config.INPUT_SIZE,
        num_classes=config.NUM_CLASSES,
        augment=False,  # No augmentation for validation
        normalize=True
    )
    
    # Quick test mode: use subset
    if config.QUICK_TEST:
        train_indices = list(range(min(config.QUICK_TEST_SAMPLES, len(train_dataset))))
        val_indices = list(range(min(config.QUICK_TEST_SAMPLES // 2, len(val_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        print(f"Quick test mode: using {len(train_dataset)} train and {len(val_dataset)} val samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader


def create_loss_function(config):
    """Create loss function based on config."""
    
    if config.LOSS_TYPE == "bce":
        pos_weight = torch.tensor([config.POS_WEIGHT]) if config.POS_WEIGHT else None
        return FOMOLoss(pos_weight=pos_weight)
    
    elif config.LOSS_TYPE == "focal":
        return FocalFOMOLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    
    elif config.LOSS_TYPE == "weighted":
        # For weighted loss, you'd need to specify class weights in config
        return WeightedFOMOLoss()
    
    else:
        raise ValueError(f"Unknown loss type: {config.LOSS_TYPE}")


def create_optimizer(model, config):
    """Create optimizer based on config."""
    
    if config.OPTIMIZER == "adam":
        return Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")


def create_scheduler(optimizer, config, num_epochs):
    """Create learning rate scheduler based on config."""
    
    if not config.USE_SCHEDULER:
        return None
    
    if config.SCHEDULER_TYPE == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=config.MIN_LR
        )
    elif config.SCHEDULER_TYPE == "step":
        return StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler type: {config.SCHEDULER_TYPE}")


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
    
    for batch_idx, (images, heatmaps) in enumerate(pbar):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        
        # Forward pass (get logits for BCEWithLogitsLoss)
        logits = model(images, return_logits=True)
        
        # Compute loss
        loss = criterion(logits, heatmaps)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, device, epoch, config):
    """Validate the model."""
    
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]  ")
    
    with torch.no_grad():
        for batch_idx, (images, heatmaps) in enumerate(pbar):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            
            # Forward pass
            logits = model(images, return_logits=True)
            
            # Compute loss
            loss = criterion(logits, heatmaps)
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            pbar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, val_loss, config, filename):
    """Save model checkpoint."""
    
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': {
            'input_size': config.INPUT_SIZE,
            'num_classes': config.NUM_CLASSES,
            'output_size': config.OUTPUT_SIZE
        }
    }
    
    filepath = checkpoint_dir / filename
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    
    print(f"✓ Checkpoint loaded from epoch {epoch}, val_loss: {val_loss:.4f}")
    return epoch, val_loss


def train(config):
    """Main training function."""
    
    print("=" * 70)
    print("FOMO-like YOLOv8 Training")
    print("=" * 70)
    
    # Print configuration
    config.print_config()
    
    # Set device
    device = torch.device(config.DEVICE)
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches:   {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = MicroYOLO(nc=config.NUM_CLASSES, input_size=config.INPUT_SIZE)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Create loss, optimizer, scheduler
    criterion = create_loss_function(config)
    criterion = criterion.to(device)  # Move loss to device (for pos_weight tensor)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, config.NUM_EPOCHS)
    
    print(f"✓ Loss function: {config.LOSS_TYPE}")
    print(f"✓ Optimizer: {config.OPTIMIZER}")
    print(f"✓ Scheduler: {config.SCHEDULER_TYPE if config.USE_SCHEDULER else 'None'}")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Validate
        if (epoch + 1) % config.VAL_FREQUENCY == 0:
            val_loss = validate(model, val_loader, criterion, device, epoch, config)
        else:
            val_loss = None
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = config.LEARNING_RATE
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        if val_loss is not None:
            print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        
        # Save checkpoint
        should_save = False
        
        if val_loss is not None:
            if config.SAVE_BEST_ONLY:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    should_save = True
                    print(f"  ✓ New best validation loss!")
            else:
                if (epoch + 1) % config.SAVE_FREQUENCY == 0:
                    should_save = True
        
        if should_save:
            filename = f"best_model.pth" if config.SAVE_BEST_ONLY else f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, config, filename)
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    save_checkpoint(model, optimizer, config.NUM_EPOCHS - 1, val_loss, config, "final_model.pth")


def main():
    parser = argparse.ArgumentParser(description='Train FOMO-like YOLOv8 model')
    
    # Model arguments
    parser.add_argument('--input-size', type=int, help='Input image size')
    parser.add_argument('--num-classes', type=int, help='Number of classes')
    
    # Dataset arguments
    parser.add_argument('--train-image-dir', type=str, help='Training images directory')
    parser.add_argument('--train-label-dir', type=str, help='Training labels directory')
    parser.add_argument('--val-image-dir', type=str, help='Validation images directory')
    parser.add_argument('--val-label-dir', type=str, help='Validation labels directory')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    # Other arguments
    parser.add_argument('--quick-test', action='store_true', help='Quick test with small dataset')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Update config from args
    Config.from_args(args)
    
    # Train
    train(Config)


if __name__ == "__main__":
    main()
