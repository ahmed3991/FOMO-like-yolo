"""
Download Kaggle Grocery Items Dataset Script

CLI wrapper for downloading Kaggle Grocery Items dataset using src.datasets.download module.

Usage:
    # Setup credentials
    python scripts/download_kaggle_grocery.py --setup
    
    # Download and prepare
    python scripts/download_kaggle_grocery.py --download --prepare
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.download import (
    setup_kaggle_credentials,
    download_kaggle_grocery,
    extract_and_prepare_grocery
)


def main():
    parser = argparse.ArgumentParser(description='Download Kaggle Grocery Items dataset')
    
    parser.add_argument('--setup', action='store_true',
                       help='Setup Kaggle API credentials')
    parser.add_argument('--download', action='store_true',
                       help='Download dataset from Kaggle')
    parser.add_argument('--prepare', action='store_true',
                       help='Extract and prepare dataset')
    parser.add_argument('--data-dir', type=str, default='data/grocery_items',
                       help='Output directory for dataset')
    
    args = parser.parse_args()
    
    # Show help if no action
    if not any([args.setup, args.download, args.prepare]):
        parser.print_help()
        print("\nExample usage:")
        print("  # Setup credentials")
        print("  python scripts/download_kaggle_grocery.py --setup")
        print("\n  # Download and prepare")
        print("  python scripts/download_kaggle_grocery.py --download --prepare")
        return
    
    # Setup credentials
    if args.setup:
        setup_kaggle_credentials()
        return
    
    # Download dataset
    if args.download:
        # Check credentials first
        if not setup_kaggle_credentials():
            print("\n✗ Please setup Kaggle credentials first")
            print("  Run: python scripts/download_kaggle_grocery.py --setup")
            return
        
        success = download_kaggle_grocery(args.data_dir)
        if not success:
            return
    
    # Prepare dataset
    if args.prepare:
        success = extract_and_prepare_grocery(args.data_dir)
        if not success:
            return
    
    # Print final instructions
    if args.download or args.prepare:
        data_path = Path(args.data_dir)
        num_classes_file = data_path / "num_classes.txt"
        
        num_classes = 2  # default
        if num_classes_file.exists():
            with open(num_classes_file, 'r') as f:
                num_classes = int(f.read().strip())
        
        print("\n" + "=" * 70)
        print("Dataset Ready for Training!")
        print("=" * 70)
        
        # Find the actual train directory
        train_options = [
            data_path / "train",
            data_path / "Starter_Dataset" / "train",
        ]
        
        train_dir = None
        for opt in train_options:
            if (opt / "images").exists():
                train_dir = opt
                break
        
        if train_dir:
            print(f"\nDataset location: {data_path}")
            print(f"  Train: {train_dir}")
            print(f"  Valid: {train_dir.parent / 'valid'}")
            print(f"  Classes: {num_classes}")
            
            print("\n Next step - Transfer Learning:")
            print(f"""
python transfer_learning.py \\
  --checkpoint checkpoints/best_model.pth \\
  --train-image-dir {train_dir}/images \\
  --train-label-dir {train_dir}/labels \\
  --val-image-dir {train_dir.parent}/valid/images \\
  --val-label-dir {train_dir.parent}/valid/labels \\
  --num-classes {num_classes} \\
  --epochs 20
""")


if __name__ == "__main__":
    main()
