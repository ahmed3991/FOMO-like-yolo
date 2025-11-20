"""
Download Full COCO 2017 Dataset Script

CLI wrapper for downloading the full COCO 2017 dataset.
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.download.coco import download_coco_full
from src.datasets.download.utils import update_config_paths, verify_dataset


def main():
    parser = argparse.ArgumentParser(description='Download Full COCO 2017 dataset')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory to store dataset (default: data)')
    parser.add_argument('--update-config', action='store_true',
                       help='Update config.py with dataset paths')
    parser.add_argument('--verify', action='store_true',
                       help='Verify dataset after download')
    
    args = parser.parse_args()
    
    # Download dataset
    dataset_dir = download_coco_full(args.data_dir)
    
    if dataset_dir:
        train_img = f"{dataset_dir}/train/images"
        train_lbl = f"{dataset_dir}/train/labels"
        val_img = f"{dataset_dir}/val/images"
        val_lbl = f"{dataset_dir}/val/labels"
        
        # Verify if requested
        if args.verify:
            print("\n" + "=" * 70)
            verify_dataset(train_img, train_lbl)
            verify_dataset(val_img, val_lbl)
        
        # Update config if requested
        if args.update_config:
            update_config_paths(train_img, train_lbl, val_img, val_lbl)
        
        # Print summary
        print("\n" + "=" * 70)
        print("Dataset Setup Complete!")
        print("=" * 70)
        print(f"\nDataset paths:")
        print(f"  Train images: {train_img}")
        print(f"  Train labels: {train_lbl}")
        print(f"  Val images:   {val_img}")
        print(f"  Val labels:   {val_lbl}")
        
        if not args.update_config:
            print("\n💡 Tip: Run with --update-config to automatically update config.py")
        
        print("\n🚀 Ready to train!")
        print("  Run: python scripts/train_coco.py")


if __name__ == "__main__":
    main()
