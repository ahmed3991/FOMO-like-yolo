"""
Prediction Visualization Script

Generate predictions on test images and create visualizations.

Usage:
    python visualize_predictions.py \
        --checkpoint checkpoints_transfer/best_model_transfer.pth \
        --image-dir data/beer_cans/test/images \
        --label-dir data/beer_cans/test/labels \
        --num-samples 10 \
        --output-dir predictions
"""

import os
import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models import MicroYOLO
from src.datasets import load_yolo_labels, yolo_to_heatmap, get_image_label_pairs
from src.evaluation import get_centroids_from_heatmap, visualize_predictions


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    num_classes = checkpoint['config']['num_classes']
    input_size = checkpoint['config']['input_size']
    
    model = MicroYOLO(nc=num_classes, input_size=input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    print(f"  Classes: {num_classes}")
    print(f"  Input size: {input_size}×{input_size}")
    
    return model, num_classes, input_size


def preprocess_image(image_path, input_size):
    """Preprocess image for model inference."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Store original for visualization
    original = image.copy()
    
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
    
    return tensor, original, resized


def create_prediction_grid(images, predictions, gt_heatmaps, class_names, threshold=0.5):
    """
    Create a grid visualization of predictions.
    
    Args:
        images: List of original images
        predictions: List of predicted heatmaps
        gt_heatmaps: List of ground truth heatmaps
        class_names: List of class names
        threshold: Detection threshold
        
    Returns:
        Grid image as numpy array
    """
    num_images = len(images)
    cols = min(5, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(num_images):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        image = images[idx]
        pred_heatmap = predictions[idx]
        gt_heatmap = gt_heatmaps[idx] if gt_heatmaps else None
        
        # Get centroids
        pred_centroids = get_centroids_from_heatmap(pred_heatmap, threshold)
        
        # Draw predictions on image
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        for class_id, cx, cy in pred_centroids:
            x = int(cx * w)
            y = int(cy * h)
            color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
            cv2.circle(vis_image, (x, y), 8, color, -1)
            cv2.circle(vis_image, (x, y), 12, color, 2)
            
            # Add label
            label = class_names[class_id] if class_names else f'C{class_id}'
            cv2.putText(vis_image, label, (x+15, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Convert BGR to RGB for matplotlib
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        ax.imshow(vis_image_rgb)
        ax.set_title(f'Image {idx+1}: {len(pred_centroids)} detections')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert figure to numpy array (compatible with newer matplotlib)
    fig.canvas.draw()
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    buf = fig.canvas.buffer_rgba()
    grid_image = np.asarray(buf)
    grid_image = grid_image[:, :, :3]  # Remove alpha channel
    
    plt.close()
    
    return grid_image


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay heatmap on image.
    
    Args:
        image: Original image (H, W, 3)
        heatmap: Heatmap (H, W) or (C, H, W)
        alpha: Opacity of heatmap
        
    Returns:
        Image with heatmap overlay
    """
    # If heatmap has channels, take max across channels
    if len(heatmap.shape) == 3:
        heatmap = np.max(heatmap, axis=0)
    
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize to 0-255
    heatmap_norm = (heatmap_resized * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # Blend
    overlay = cv2.addWeighted(image, 1-alpha, heatmap_color, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Directory containing test images')
    
    # Optional arguments
    parser.add_argument('--label-dir', type=str,
                       help='Directory containing ground truth labels (optional)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='predictions',
                       help='Output directory for visualizations')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold')
    parser.add_argument('--class-names', type=str, nargs='+',
                       help='Class names (e.g., --class-names beer can)')
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual prediction images')
    parser.add_argument('--create-grid', action='store_true', default=True,
                       help='Create grid visualization')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FOMO Prediction Visualization")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading model...")
    model, num_classes, input_size = load_model(args.checkpoint, device)
    
    # Get image-label pairs
    print("\nLoading images...")
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir) if args.label_dir else None
    
    if label_dir:
        pairs = get_image_label_pairs(image_dir, label_dir)
    else:
        # No labels provided, just get images
        pairs = [(str(p), None) for p in image_dir.glob("*.jpg")] + \
                [(str(p), None) for p in image_dir.glob("*.png")]
    
    print(f"✓ Found {len(pairs)} images")
    
    # Limit to num_samples
    pairs = pairs[:args.num_samples]
    print(f"  Processing {len(pairs)} samples")
    
    # Generate predictions
    print("\nGenerating predictions...")
    
    images_list = []
    predictions_list = []
    gt_heatmaps_list = []
    
    for img_path, label_path in tqdm(pairs):
        # Preprocess image
        image_tensor, original, resized = preprocess_image(img_path, input_size)
        image_tensor = image_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            pred_heatmap = model(image_tensor)
        
        pred_heatmap = pred_heatmap[0].cpu()
        
        # Load ground truth if available
        if label_path:
            labels = load_yolo_labels(label_path)
            gt_heatmap = yolo_to_heatmap(labels, input_size // 8, num_classes)
        else:
            gt_heatmap = None
        
        images_list.append(original)
        predictions_list.append(pred_heatmap)
        gt_heatmaps_list.append(gt_heatmap)
        
        # Save individual visualization
        if args.save_individual:
            img_name = Path(img_path).stem
            save_path = output_dir / f"{img_name}_pred.png"
            
            if gt_heatmap is not None:
                visualize_predictions(
                    resized, pred_heatmap, gt_heatmap,
                    threshold=args.threshold,
                    class_names=args.class_names,
                    save_path=str(save_path)
                )
            else:
                # Visualize predictions with heatmap overlay
                vis_image = original.copy()
                h, w = vis_image.shape[:2]
                
                # Overlay heatmap
                vis_image = overlay_heatmap(vis_image, pred_heatmap.numpy(), alpha=0.4)
                
                pred_centroids = get_centroids_from_heatmap(pred_heatmap, args.threshold)
                
                for class_id, cx, cy in pred_centroids:
                    x = int(cx * w)
                    y = int(cy * h)
                    color = (0, 255, 0)
                    cv2.circle(vis_image, (x, y), 8, color, -1)
                    cv2.circle(vis_image, (x, y), 12, color, 2)
                    
                    # Add label and coordinates
                    label = args.class_names[class_id] if args.class_names else f'C{class_id}'
                    text = f"{label}: ({x}, {y})"
                    cv2.putText(vis_image, text, (x+15, y+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imwrite(str(save_path), vis_image)
    
    # Create grid visualization
    if args.create_grid:
        print("\nCreating grid visualization...")
        grid_image = create_prediction_grid(
            images_list,
            predictions_list,
            gt_heatmaps_list if label_dir else None,
            args.class_names,
            args.threshold
        )
        
        grid_path = output_dir / "predictions_grid.png"
        cv2.imwrite(str(grid_path), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
        print(f"✓ Grid saved to: {grid_path}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Prediction Statistics")
    print("=" * 70)
    
    total_detections = 0
    for pred_heatmap in predictions_list:
        centroids = get_centroids_from_heatmap(pred_heatmap, args.threshold)
        total_detections += len(centroids)
    
    print(f"Total images: {len(images_list)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / len(images_list):.2f}")
    print(f"\nVisualization saved to: {output_dir}/")
    
    if args.save_individual:
        print(f"  - Individual predictions: {len(pairs)} files")
    if args.create_grid:
        print(f"  - Grid visualization: predictions_grid.png")


if __name__ == "__main__":
    main()
