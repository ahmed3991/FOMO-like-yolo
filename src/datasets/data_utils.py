"""
Data Utilities for FOMO-like YOLOv8 Model

Functions for converting YOLO bounding box labels to center-point heatmaps
and various data processing utilities.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt


def load_yolo_labels(label_path):
    """
    Load YOLO format labels from a text file.
    
    Args:
        label_path: Path to YOLO .txt label file
        
    Returns:
        List of (class_id, cx, cy, w, h) tuples (all normalized 0-1)
        Returns empty list if file doesn't exist or is empty
    """
    if not os.path.exists(label_path):
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                labels.append((class_id, cx, cy, w, h))
    
    return labels


def yolo_to_heatmap(labels, output_size, num_classes):
    """
    Convert YOLO bounding box labels to binary center-point heatmaps.
    
    Args:
        labels: List of (class_id, cx, cy, w, h) tuples (normalized 0-1)
        output_size: Size of the output heatmap (e.g., 20 for 20x20)
        num_classes: Number of object classes
        
    Returns:
        Tensor of shape (num_classes, output_size, output_size) with 1s at object centers
    """
    heatmap = torch.zeros(num_classes, output_size, output_size, dtype=torch.float32)
    
    for class_id, cx, cy, w, h in labels:
        # Convert normalized coordinates to grid indices
        # cx, cy are in [0, 1], we need to map to [0, output_size-1]
        grid_x = int(cx * output_size)
        grid_y = int(cy * output_size)
        
        # Clamp to valid range
        grid_x = max(0, min(output_size - 1, grid_x))
        grid_y = max(0, min(output_size - 1, grid_y))
        
        # Ensure class_id is valid
        if 0 <= class_id < num_classes:
            heatmap[class_id, grid_y, grid_x] = 1.0
    
    return heatmap


def visualize_heatmap(image, heatmap, class_names=None, save_path=None, alpha=0.6):
    """
    Visualize heatmap overlaid on image for debugging.
    
    Args:
        image: Input image (numpy array, uint8, RGB or BGR)
        heatmap: Heatmap tensor of shape (num_classes, H, W)
        class_names: Optional list of class names for legend
        save_path: If provided, save visualization to this path
        alpha: Transparency of heatmap overlay (0-1)
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy()
    
    # Convert image to RGB if needed
    if image.shape[2] == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = image.copy()
    
    num_classes = heatmap.shape[0]
    h, w = heatmap.shape[1], heatmap.shape[2]
    
    # Resize heatmap to match image size
    img_h, img_w = display_image.shape[:2]
    
    # Create figure
    fig, axes = plt.subplots(1, num_classes + 1, figsize=(5 * (num_classes + 1), 5))
    if num_classes == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
    
    # Show original image
    axes[0].imshow(display_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show heatmap for each class
    colors = plt.cm.get_cmap('tab10', num_classes)
    
    for class_id in range(num_classes):
        class_heatmap = heatmap[class_id]
        
        # Resize heatmap to image size
        resized_heatmap = cv2.resize(class_heatmap, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # Create colored heatmap
        colored_heatmap = np.zeros((img_h, img_w, 3))
        color = colors(class_id)[:3]
        for i in range(3):
            colored_heatmap[:, :, i] = resized_heatmap * color[i]
        
        # Overlay on image
        overlay = display_image.copy().astype(float) / 255.0
        mask = resized_heatmap > 0
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * colored_heatmap[mask]
        
        class_name = class_names[class_id] if class_names else f'Class {class_id}'
        axes[class_id + 1].imshow(overlay)
        axes[class_id + 1].set_title(f'{class_name} Heatmap')
        axes[class_id + 1].axis('off')
        
        # Mark center points
        center_points = np.argwhere(class_heatmap > 0)
        for pt in center_points:
            y, x = pt
            # Scale to image coordinates
            img_x = int(x * img_w / w)
            img_y = int(y * img_h / h)
            axes[class_id + 1].plot(img_x, img_y, 'r*', markersize=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def get_image_label_pairs(image_dir, label_dir, image_extensions=None):
    """
    Find matching image and label file pairs.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing YOLO label files (.txt)
        image_extensions: List of valid image extensions (default: ['.jpg', '.jpeg', '.png'])
        
    Returns:
        List of (image_path, label_path) tuples
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    pairs = []
    
    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() in image_extensions:
            # Find corresponding label file
            label_path = label_dir / (img_path.stem + '.txt')
            if label_path.exists():
                pairs.append((str(img_path), str(label_path)))
    
    return pairs


if __name__ == "__main__":
    """
    Demonstration and testing of data utilities.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FOMO data utilities')
    parser.add_argument('--visualize', action='store_true', help='Run visualization demo')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--image-dir', type=str, help='Image directory for visualization')
    parser.add_argument('--label-dir', type=str, help='Label directory for visualization')
    parser.add_argument('--output-dir', type=str, default='./debug_output', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if args.visualize:
        if not args.image_dir or not args.label_dir:
            print("Error: --image-dir and --label-dir are required for visualization")
            exit(1)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Get image-label pairs
        pairs = get_image_label_pairs(args.image_dir, args.label_dir)
        
        if not pairs:
            print(f"No matching image-label pairs found in {args.image_dir} and {args.label_dir}")
            exit(1)
        
        print(f"Found {len(pairs)} image-label pairs")
        
        # Visualize samples
        for i, (img_path, label_path) in enumerate(pairs[:args.num_samples]):
            print(f"\nProcessing sample {i+1}/{min(args.num_samples, len(pairs))}: {os.path.basename(img_path)}")
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"  Error loading image: {img_path}")
                continue
            
            # Load labels
            labels = load_yolo_labels(label_path)
            print(f"  Found {len(labels)} objects")
            
            # Generate heatmap (assume 2 classes, output size 20)
            heatmap = yolo_to_heatmap(labels, output_size=20, num_classes=2)
            
            # Visualize
            save_path = os.path.join(args.output_dir, f'sample_{i+1}.png')
            visualize_heatmap(image, heatmap, class_names=['Class 0', 'Class 1'], save_path=save_path)
        
        print(f"\n✓ Visualization complete! Check {args.output_dir} for results")
    
    else:
        # Run basic tests
        print("Running basic tests...")
        
        # Test 1: Empty labels
        heatmap = yolo_to_heatmap([], output_size=20, num_classes=2)
        assert heatmap.shape == (2, 20, 20)
        assert heatmap.sum() == 0
        print("✓ Test 1 passed: Empty labels")
        
        # Test 2: Single object
        labels = [(0, 0.5, 0.5, 0.1, 0.1)]  # Class 0 at center
        heatmap = yolo_to_heatmap(labels, output_size=20, num_classes=2)
        assert heatmap.shape == (2, 20, 20)
        assert heatmap[0, 10, 10] == 1.0  # Center point marked
        assert heatmap[1].sum() == 0  # Class 1 empty
        print("✓ Test 2 passed: Single object")
        
        # Test 3: Multiple objects
        labels = [
            (0, 0.25, 0.25, 0.1, 0.1),
            (1, 0.75, 0.75, 0.1, 0.1),
            (0, 0.5, 0.5, 0.1, 0.1)
        ]
        heatmap = yolo_to_heatmap(labels, output_size=20, num_classes=2)
        assert heatmap[0].sum() == 2.0  # 2 objects of class 0
        assert heatmap[1].sum() == 1.0  # 1 object of class 1
        print("✓ Test 3 passed: Multiple objects")
        
        # Test 4: Boundary coordinates
        labels = [(0, 0.0, 0.0, 0.1, 0.1), (1, 1.0, 1.0, 0.1, 0.1)]
        heatmap = yolo_to_heatmap(labels, output_size=20, num_classes=2)
        assert heatmap[0, 0, 0] == 1.0  # Top-left corner
        assert heatmap[1, 19, 19] == 1.0  # Bottom-right corner
        print("✓ Test 4 passed: Boundary coordinates")
        
        print("\n✓ All tests passed!")
