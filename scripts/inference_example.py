"""
Quick Start Example for FOMO-like YOLOv8

This script demonstrates basic usage of the model for inference.
"""

import sys
import torch
import cv2
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import MicroYOLO
from src.evaluation import get_centroids_from_heatmap, visualize_predictions
from src.datasets import yolo_to_heatmap, load_yolo_labels


def load_model(checkpoint_path, num_classes=2, input_size=160, device='cpu'):
    """Load trained model from checkpoint."""
    model = MicroYOLO(nc=num_classes, input_size=input_size)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("  Using untrained model")
    
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, input_size=160):
    """Load and preprocess image for inference."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize
    resized = cv2.resize(image, (input_size, input_size))
    
    # Convert to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize
    normalized = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (normalized - mean) / std
    
    # HWC to CHW
    transposed = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension and convert to tensor
    tensor = torch.from_numpy(transposed).unsqueeze(0).float()
    
    return tensor, image


def inference_example():
    """Example inference workflow."""
    print("=" * 70)
    print("FOMO-like YOLOv8 Inference Example")
    print("=" * 70)
    
    # Configuration
    checkpoint_path = "checkpoints/best_model.pth"
    input_size = 160
    num_classes = 2
    threshold = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    print(f"Input size: {input_size}×{input_size}")
    print(f"Detection threshold: {threshold}")
    
    # Load model
    print("\n[1] Loading model...")
    model = load_model(checkpoint_path, num_classes, input_size, device)
    
    # NOTE: Since we don't have actual data yet, this is just a demonstration
    print("\n[2] Inference demonstration...")
    print("    (Using random data - replace with actual image in practice)")
    
    # Create dummy input for demo
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Inference
    with torch.no_grad():
        heatmap = model(dummy_input)  # Returns probabilities (with sigmoid)
    
    print(f"    Output shape: {heatmap.shape}")
    print(f"    Output range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Extract centroids
    print("\n[3] Extracting object centroids...")
    centroids = get_centroids_from_heatmap(heatmap[0], threshold=threshold)
    
    print(f"    Detected {len(centroids)} objects:")
    for i, (class_id, cx, cy) in enumerate(centroids):
        print(f"      Object {i+1}: Class {class_id}, Center ({cx:.3f}, {cy:.3f})")
    
    print("\n" + "=" * 70)
    print("✓ Inference example complete!")
    print("=" * 70)
    print("\nTo use with actual images:")
    print("  1. Train the model (see README.md)")
    print("  2. Replace dummy_input with preprocessed real image")
    print("  3. Visualize results with visualize_predictions()")


def real_image_inference_example(image_path, label_path=None):
    """
    Example with real image (requires trained model and actual data).
    
    Args:
        image_path: Path to input image
        label_path: Optional path to ground truth labels for comparison
    """
    checkpoint_path = "checkpoints/best_model.pth"
    input_size = 160
    num_classes = 2
    threshold = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load model
    model = load_model(checkpoint_path, num_classes, input_size, device)
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path, input_size)
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        heatmap = model(image_tensor)
    
    # Extract centroids
    centroids = get_centroids_from_heatmap(heatmap[0], threshold=threshold)
    
    print(f"Detected {len(centroids)} objects in {Path(image_path).name}")
    
    # If ground truth provided, compare
    if label_path:
        labels = load_yolo_labels(label_path)
        gt_heatmap = yolo_to_heatmap(labels, input_size // 8, num_classes)
        
        # Visualize
        visualize_predictions(
            original_image,
            heatmap[0],
            gt_heatmap,
            threshold=threshold,
            class_names=[f'Class {i}' for i in range(num_classes)],
            save_path='prediction_result.png'
        )
        print("Visualization saved to: prediction_result.png")
    
    return centroids


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FOMO inference example')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--label', type=str, help='Path to ground truth label (optional)')
    
    args = parser.parse_args()
    
    if args.image:
        # Real inference
        centroids = real_image_inference_example(args.image, args.label)
    else:
        # Demo with dummy data
        inference_example()
