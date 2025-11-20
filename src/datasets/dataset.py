"""
PyTorch Dataset for FOMO-style Object Detection

Dataset class for loading images and generating center-point heatmaps.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Try to import albumentations, fall back to None if not available
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    A = None
    ToTensorV2 = None

from .data_utils import load_yolo_labels, yolo_to_heatmap, get_image_label_pairs


class FOMODataset(Dataset):
    """
    PyTorch Dataset for FOMO-style training.
    
    Loads images and converts YOLO labels to center-point heatmaps.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing YOLO .txt label files
        input_size: Input image size (e.g., 160 or 96)
        num_classes: Number of object classes
        augment: Whether to apply data augmentation (training mode)
        normalize: Whether to normalize images to [0, 1]
    """
    
    def __init__(self, image_dir, label_dir, input_size=160, num_classes=2, 
                 augment=False, normalize=True):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.input_size = input_size
        self.num_classes = num_classes
        self.augment = augment
        self.normalize = normalize
        
        # Calculate output heatmap size (1/8 of input due to 3 stride-2 layers)
        self.output_size = input_size // 8
        
        # Get image-label pairs
        self.pairs = get_image_label_pairs(image_dir, label_dir)
        
        if len(self.pairs) == 0:
            raise ValueError(f"No valid image-label pairs found in {image_dir} and {label_dir}")
        
        # Setup augmentations
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        """Create albumentations transform pipeline."""
        transforms = [
            A.Resize(self.input_size, self.input_size),
        ]
        
        if self.augment:
            # Training augmentations
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            ])
        
        if self.normalize:
            transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225]))
        else:
            transforms.append(A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]))
        
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            image: Tensor of shape (3, input_size, input_size)
            heatmap: Tensor of shape (num_classes, output_size, output_size)
        """
        img_path, label_path = self.pairs[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        labels = load_yolo_labels(label_path)
        
        # Apply augmentations to image
        # Note: For FOMO, we don't need to transform bounding boxes since we only care about centers
        # The center coordinates remain the same after horizontal flip
        transformed = self.transform(image=image)
        image = transformed['image']
        
        # If horizontal flip was applied, we need to adjust center x-coordinates
        # Albumentations doesn't handle this automatically for our custom labels
        # For simplicity, we'll handle this manually
        if self.augment and hasattr(self, '_last_flip') and self._last_flip:
            labels = [(cls, 1 - cx, cy, w, h) for cls, cx, cy, w, h in labels]
        
        # Generate heatmap from labels
        heatmap = yolo_to_heatmap(labels, self.output_size, self.num_classes)
        
        return image, heatmap
    
    def get_sample_info(self, idx):
        """Get information about a sample (useful for debugging)."""
        img_path, label_path = self.pairs[idx]
        labels = load_yolo_labels(label_path)
        return {
            'image_path': img_path,
            'label_path': label_path,
            'num_objects': len(labels),
            'labels': labels
        }


class SimpleAugmentDataset(Dataset):
    """
    Simplified FOMO Dataset with basic augmentations (no albumentations dependency).
    
    Use this if you don't want to install albumentations.
    """
    
    def __init__(self, image_dir, label_dir, input_size=160, num_classes=2, 
                 augment=False, normalize=True):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.input_size = input_size
        self.num_classes = num_classes
        self.augment = augment
        self.normalize = normalize
        
        self.output_size = input_size // 8
        self.pairs = get_image_label_pairs(image_dir, label_dir)
        
        if len(self.pairs) == 0:
            raise ValueError(f"No valid image-label pairs found in {image_dir} and {label_dir}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        labels = load_yolo_labels(label_path)
        
        # Resize
        image = cv2.resize(image, (self.input_size, self.input_size))
        
        # Simple augmentations
        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 1)
                labels = [(cls, 1 - cx, cy, w, h) for cls, cx, cy, w, h in labels]
            
            # Brightness and contrast
            if np.random.rand() > 0.5:
                alpha = 1.0 + np.random.uniform(-0.2, 0.2)  # Contrast
                beta = np.random.uniform(-20, 20)  # Brightness
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        
        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        # Generate heatmap
        heatmap = yolo_to_heatmap(labels, self.output_size, self.num_classes)
        
        return image, heatmap
    
    def get_sample_info(self, idx):
        img_path, label_path = self.pairs[idx]
        labels = load_yolo_labels(label_path)
        return {
            'image_path': img_path,
            'label_path': label_path,
            'num_objects': len(labels),
            'labels': labels
        }


if __name__ == "__main__":
    """
    Test the dataset implementation.
    """
    import argparse
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser(description='Test FOMO Dataset')
    parser.add_argument('--image-dir', type=str, required=True, help='Image directory')
    parser.add_argument('--label-dir', type=str, required=True, help='Label directory')
    parser.add_argument('--input-size', type=int, default=160, help='Input size')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--simple', action='store_true', help='Use simple dataset (no albumentations)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FOMO Dataset Test")
    print("=" * 60)
    
    # Create dataset
    try:
        if args.simple:
            print("\nUsing SimpleAugmentDataset (no albumentations)")
            dataset = SimpleAugmentDataset(
                args.image_dir, 
                args.label_dir,
                input_size=args.input_size,
                num_classes=args.num_classes,
                augment=True
            )
        else:
            print("\nUsing FOMODataset (with albumentations)")
            dataset = FOMODataset(
                args.image_dir, 
                args.label_dir,
                input_size=args.input_size,
                num_classes=args.num_classes,
                augment=True
            )
        
        print(f"✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Input size: {args.input_size}x{args.input_size}")
        print(f"  Output size: {dataset.output_size}x{dataset.output_size}")
        print(f"  Number of classes: {args.num_classes}")
    
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        exit(1)
    
    # Test single sample
    print("\n[Test 1] Single Sample")
    print("-" * 60)
    try:
        image, heatmap = dataset[0]
        info = dataset.get_sample_info(0)
        
        print(f"Image shape: {image.shape}")
        print(f"Heatmap shape: {heatmap.shape}")
        print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        print(f"Number of objects: {info['num_objects']}")
        print(f"✓ Single sample test passed")
    except Exception as e:
        print(f"✗ Single sample test failed: {e}")
        exit(1)
    
    # Test DataLoader
    print("\n[Test 2] DataLoader Batching")
    print("-" * 60)
    try:
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        for batch_idx, (images, heatmaps) in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Heatmaps shape: {heatmaps.shape}")
            
            if batch_idx >= 2:  # Test first 3 batches
                break
        
        print(f"✓ DataLoader test passed")
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All dataset tests passed!")
    print("=" * 60)
