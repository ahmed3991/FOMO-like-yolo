"""
Common download utilities for dataset preparation.

Contains helper functions for downloading files, extracting archives,
creating train/val splits, and managing dataset paths.
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import random


def download_file(url, dest_path):
    """Download file with progress bar."""
    print(f"Downloading from {url}...")
    
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}%", end='')
    
    urllib.request.urlretrieve(url, dest_path, reporthook)
    print("\n✓ Download complete")


def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✓ Extraction complete")


def create_train_val_split(images_dir, labels_dir, output_dir, split_ratio=0.8):
    """
    Create train/val split from a single directory.
    
    Args:
        images_dir: Source images directory
        labels_dir: Source labels directory
        output_dir: Output directory for train/val split
        split_ratio: Ratio of training samples (default: 0.8)
        
    Returns:
        Tuple of (train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir)
    """
    print(f"\nCreating train/val split ({int(split_ratio*100)}/{int((1-split_ratio)*100)})...")
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    # Get all images
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    
    # Shuffle
    random.seed(42)
    random.shuffle(image_files)
    
    # Split
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Create directories
    train_img_dir = output_dir / "train" / "images"
    train_lbl_dir = output_dir / "train" / "labels"
    val_img_dir = output_dir / "val" / "images"
    val_lbl_dir = output_dir / "val" / "labels"
    
    for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    def copy_files(file_list, img_dest, lbl_dest):
        for img_file in file_list:
            # Copy image
            shutil.copy2(img_file, img_dest / img_file.name)
            
            # Copy label if exists
            lbl_file = labels_dir / (img_file.stem + ".txt")
            if lbl_file.exists():
                shutil.copy2(lbl_file, lbl_dest / lbl_file.name)
    
    print("  Copying training files...")
    copy_files(train_files, train_img_dir, train_lbl_dir)
    
    print("  Copying validation files...")
    copy_files(val_files, val_img_dir, val_lbl_dir)
    
    print(f"\n✓ Split complete!")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val:   {len(val_files)} images")
    print(f"  Location: {output_dir}")
    
    return str(train_img_dir), str(train_lbl_dir), str(val_img_dir), str(val_lbl_dir)


def verify_dataset(image_dir, label_dir):
    """Verify dataset structure and content."""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    print("\nVerifying dataset...")
    
    if not image_dir.exists():
        print(f"✗ Image directory not found: {image_dir}")
        return False
    
    if not label_dir.exists():
        print(f"✗ Label directory not found: {label_dir}")
        return False
    
    # Count files
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    labels = list(label_dir.glob("*.txt"))
    
    print(f"  Images: {len(images)}")
    print(f"  Labels: {len(labels)}")
    
    # Check for matching files
    matched = 0
    for img in images[:10]:  # Check first 10
        lbl = label_dir / (img.stem + ".txt")
        if lbl.exists():
            matched += 1
    
    print(f"  Sample match rate: {matched}/10")
    
    if matched > 0:
        print("✓ Dataset verification passed")
        return True
    else:
        print("✗ Dataset verification failed")
        return False


def update_config_paths(train_img, train_lbl, val_img, val_lbl, config_file="config.py"):
    """Update config.py with dataset paths."""
    print(f"\nUpdating {config_file}...")
    
    config_path = Path(config_file)
    
    # Create default config if it doesn't exist
    if not config_path.exists():
        print(f"  Creating new {config_file}...")
        default_content = """\"\"\"
Project Configuration

This file contains path configurations for datasets and training.
It is automatically updated by download scripts.
\"\"\"

# Dataset Paths
# These are updated automatically when running download scripts
TRAIN_IMAGE_DIR = "data/train/images"
TRAIN_LABEL_DIR = "data/train/labels"
VAL_IMAGE_DIR = "data/val/images"
VAL_LABEL_DIR = "data/val/labels"

# Model Configuration
INPUT_SIZE = 160
NUM_CLASSES = 80
"""
        with open(config_path, 'w') as f:
            f.write(default_content)
    
    # Read config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Update paths
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if line.startswith("TRAIN_IMAGE_DIR"):
            new_lines.append(f'TRAIN_IMAGE_DIR = "{train_img}"')
        elif line.startswith("TRAIN_LABEL_DIR"):
            new_lines.append(f'TRAIN_LABEL_DIR = "{train_lbl}"')
        elif line.startswith("VAL_IMAGE_DIR"):
            new_lines.append(f'VAL_IMAGE_DIR = "{val_img}"')
        elif line.startswith("VAL_LABEL_DIR"):
            new_lines.append(f'VAL_LABEL_DIR = "{val_lbl}"')
        else:
            new_lines.append(line)
            
    content = '\n'.join(new_lines)
    
    # Write back
    with open(config_path, 'w') as f:
        f.write(content)
    
    print("✓ Config updated successfully")
