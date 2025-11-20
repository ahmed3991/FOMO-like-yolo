"""
Kaggle Grocery Items Dataset Download Utilities

Functions for downloading and preparing the Kaggle Grocery Items dataset.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from .utils import create_train_val_split


def setup_kaggle_credentials():
    """
    Setup Kaggle API credentials.
    
    Instructions:
    1. Go to https://www.kaggle.com/settings
    2. Click "Create New API Token"
    3. Download kaggle.json
    4. Place it in ~/.kaggle/kaggle.json
    
    Returns:
        True if credentials are properly set up, False otherwise
    """
    print("=" * 70)
    print("Kaggle API Setup")
    print("=" * 70)
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print(f"✓ Kaggle credentials found: {kaggle_json}")
        
        # Set permissions
        os.chmod(kaggle_json, 0o600)
        print("✓ Permissions set correctly")
        
        return True
    else:
        print(f"✗ Kaggle credentials not found")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Download the kaggle.json file")
        print("5. Run the following command:")
        print(f"   mkdir -p {kaggle_dir}")
        print(f"   mv ~/Downloads/kaggle.json {kaggle_json}")
        print(f"   chmod 600 {kaggle_json}")
        
        return False


def download_kaggle_grocery(output_dir="data/grocery_items"):
    """
    Download Kaggle Grocery Items dataset.
    
    Args:
        output_dir: Directory to save the dataset
        
    Returns:
        True if download successful, False otherwise
    """
    print("\n" + "=" * 70)
    print("Downloading Kaggle Grocery Items Dataset")
    print("=" * 70)
    
    try:
        import kaggle
    except ImportError:
        print("✗ Error: kaggle package not installed")
        print("  Install with: pip install kaggle")
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("\nDownloading dataset (this may take a few minutes)...")
        
        # Download competition files
        # Note: You may need to accept competition rules first on Kaggle website
        kaggle.api.competition_download_files(
            'grocery-items-multi-class-object-detection',
            path=str(output_path)
        )
        
        print("✓ Download complete!")
        
        # List downloaded files
        files = list(output_path.glob("*.zip"))
        print(f"\nDownloaded files:")
        for f in files:
            print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nPossible issues:")
        print("1. You haven't accepted the competition rules")
        print("   Visit: https://www.kaggle.com/competitions/grocery-items-multi-class-object-detection")
        print("   Click 'Join Competition' and accept rules")
        print("2. Kaggle credentials not set up correctly")
        print("   Run with --setup flag to check credentials")
        
        return False


def extract_and_prepare_grocery(data_dir="data/grocery_items"):
    """
    Extract and prepare Kaggle Grocery dataset for training.
    
    Args:
        data_dir: Directory containing the downloaded dataset
        
    Returns:
        True if preparation successful, False otherwise
    """
    print("\n" + "=" * 70)
    print("Extracting and Preparing Dataset")
    print("=" * 70)
    
    data_path = Path(data_dir)
    
    # Extract all zip files
    zip_files = list(data_path.glob("*.zip"))
    
    if not zip_files:
        print("✗ No zip files found to extract")
        return False
    
    for zip_file in zip_files:
        print(f"\nExtracting {zip_file.name}...")
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        print(f"✓ Extracted {zip_file.name}")
    
    # Find the extracted directory
    # The dataset typically extracts to Starter_Dataset/
    extracted_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"\nExtracted directories: {[d.name for d in extracted_dirs]}")
    
    # Look for train/test structure
    train_dir = None
    test_dir = None
    
    for dir_path in extracted_dirs:
        if (dir_path / "train" / "images").exists():
            train_dir = dir_path / "train"
            print(f"✓ Found training data: {train_dir}")
        
        if (dir_path / "TestImages" / "images").exists():
            test_dir = dir_path / "TestImages"
            print(f"✓ Found test data: {test_dir}")
    
    if not train_dir:
        # Check if train is directly in data_path
        if (data_path / "train" / "images").exists():
            train_dir = data_path / "train"
            print(f"✓ Found training data: {train_dir}")
    
    if not train_dir:
        print("✗ Could not find training data")
        print(f"  Please check structure in {data_path}")
        return False
    
    # Count files
    train_images = len(list((train_dir / "images").glob("*.jpg"))) + \
                   len(list((train_dir / "images").glob("*.png")))
    train_labels = len(list((train_dir / "labels").glob("*.txt")))
    
    print(f"\nDataset statistics:")
    print(f"  Training images: {train_images}")
    print(f"  Training labels: {train_labels}")
    
    if test_dir:
        test_images = len(list((test_dir / "images").glob("*.jpg"))) + \
                      len(list((test_dir / "images").glob("*.png")))
        print(f"  Test images: {test_images}")
    
    # Create validation split if not exists
    val_dir = train_dir.parent / "valid"
    if not val_dir.exists():
        print("\n✓ Creating train/validation split (80/20)...")
        _create_train_val_split_inplace(
            train_dir / "images",
            train_dir / "labels",
            train_dir.parent,
            split_ratio=0.8
        )
    else:
        print("✓ Validation split already exists")
    
    # Read classes
    classes_file = None
    for dir_path in [data_path] + extracted_dirs:
        if (dir_path / "classes.txt").exists():
            classes_file = dir_path / "classes.txt"
            break
    
    if classes_file:
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        
        print(f"\n✓ Found {len(classes)} classes:")
        for i, cls in enumerate(classes[:10]):  # Show first 10
            print(f"    {i}: {cls}")
        if len(classes) > 10:
            print(f"    ... and {len(classes) - 10} more")
        
        # Save class count for later
        with open(data_path / "num_classes.txt", 'w') as f:
            f.write(str(len(classes)))
    
    print("\n✓ Dataset ready for training!")
    
    return True


def _create_train_val_split_inplace(images_dir, labels_dir, output_dir, split_ratio=0.8):
    """Create train/val split by moving files (used for Kaggle dataset)."""
    import random
    
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
    train_img_new = output_dir / "train" / "images"
    train_lbl_new = output_dir / "train" / "labels"
    val_img = output_dir / "valid" / "images"
    val_lbl = output_dir / "valid" / "labels"
    
    for dir_path in [train_img_new, train_lbl_new, val_img, val_lbl]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Move validation files
    print(f"  Moving {len(val_files)} files to validation set...")
    for img_file in val_files:
        # Move image
        shutil.move(str(img_file), str(val_img / img_file.name))
        
        # Move label if exists
        lbl_file = labels_dir / (img_file.stem + ".txt")
        if lbl_file.exists():
            shutil.move(str(lbl_file), str(val_lbl / lbl_file.name))
    
    print(f"  ✓ Train: {len(train_files)} images")
    print(f"  ✓ Val:   {len(val_files)} images")
