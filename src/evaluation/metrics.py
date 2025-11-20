"""
Evaluation utilities for FOMO-like YOLOv8 Model.

Functions for extracting centroids from heatmaps and computing metrics.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def get_centroids_from_heatmap(heatmap, threshold=0.5, min_area=1):
    """
    Extract object centroids from predicted heatmap using contour finding.
    
    Args:
        heatmap: Heatmap tensor of shape (num_classes, H, W) with values in [0, 1]
        threshold: Probability threshold for detection (default: 0.5)
        min_area: Minimum contour area to filter noise (default: 1)
        
    Returns:
        List of (class_id, x, y) tuples (normalized coordinates 0-1)
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    centroids = []
    num_classes, h, w = heatmap.shape
    
    for class_id in range(num_classes):
        # Get class-specific heatmap
        class_heatmap = heatmap[class_id]
        
        # Threshold
        binary = (class_heatmap > threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Calculate centroid using moments
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            
            # Normalize to [0, 1]
            cx_norm = cx / w
            cy_norm = cy / h
            
            centroids.append((class_id, cx_norm, cy_norm))
    
    return centroids


def match_centroids(pred_centroids, gt_centroids, distance_threshold=0.05):
    """
    Match predicted centroids to ground truth centroids.
    
    Args:
        pred_centroids: List of (class_id, x, y) predictions
        gt_centroids: List of (class_id, x, y) ground truth
        distance_threshold: Maximum normalized distance for a match (default: 0.05)
        
    Returns:
        matches: List of (pred_idx, gt_idx) tuples
        unmatched_preds: List of unmatched prediction indices
        unmatched_gts: List of unmatched ground truth indices
    """
    matches = []
    matched_preds = set()
    matched_gts = set()
    
    # For each prediction, find closest ground truth
    for pred_idx, (pred_class, pred_x, pred_y) in enumerate(pred_centroids):
        best_dist = float('inf')
        best_gt_idx = None
        
        for gt_idx, (gt_class, gt_x, gt_y) in enumerate(gt_centroids):
            # Only match same class
            if pred_class != gt_class:
                continue
            
            # Already matched
            if gt_idx in matched_gts:
                continue
            
            # Calculate Euclidean distance
            dist = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            
            if dist < distance_threshold and dist < best_dist:
                best_dist = dist
                best_gt_idx = gt_idx
        
        if best_gt_idx is not None:
            matches.append((pred_idx, best_gt_idx))
            matched_preds.add(pred_idx)
            matched_gts.add(best_gt_idx)
    
    unmatched_preds = [i for i in range(len(pred_centroids)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_centroids)) if i not in matched_gts]
    
    return matches, unmatched_preds, unmatched_gts


def evaluate_centroids(pred_centroids, gt_centroids, distance_threshold=0.05, num_classes=None):
    """
    Compute precision, recall, and F1 score for centroid predictions.
    
    Args:
        pred_centroids: List of (class_id, x, y) predictions
        gt_centroids: List of (class_id, x, y) ground truth
        distance_threshold: Maximum distance for a match
        num_classes: Number of classes (for per-class metrics)
        
    Returns:
        Dictionary with 'precision', 'recall', 'f1', and per-class metrics
    """
    matches, unmatched_preds, unmatched_gts = match_centroids(
        pred_centroids, gt_centroids, distance_threshold
    )
    
    # Overall metrics
    num_matches = len(matches)
    num_preds = len(pred_centroids)
    num_gts = len(gt_centroids)
    
    precision = num_matches / num_preds if num_preds > 0 else 0.0
    recall = num_matches / num_gts if num_gts > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_matches': num_matches,
        'num_predictions': num_preds,
        'num_ground_truth': num_gts
    }
    
    # Per-class metrics
    if num_classes is not None:
        per_class_metrics = {}
        
        for class_id in range(num_classes):
            class_preds = [c for c in pred_centroids if c[0] == class_id]
            class_gts = [c for c in gt_centroids if c[0] == class_id]
            
            if len(class_preds) == 0 and len(class_gts) == 0:
                continue
            
            class_matches, _, _ = match_centroids(class_preds, class_gts, distance_threshold)
            
            class_precision = len(class_matches) / len(class_preds) if len(class_preds) > 0 else 0.0
            class_recall = len(class_matches) / len(class_gts) if len(class_gts) > 0 else 0.0
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) \
                       if (class_precision + class_recall) > 0 else 0.0
            
            per_class_metrics[f'class_{class_id}'] = {
                'precision': class_precision,
                'recall': class_recall,
                'f1': class_f1,
                'num_predictions': len(class_preds),
                'num_ground_truth': len(class_gts)
            }
        
        metrics['per_class'] = per_class_metrics
    
    return metrics


def visualize_predictions(image, pred_heatmap, gt_heatmap, threshold=0.5, 
                         class_names=None, save_path=None):
    """
    Visualize predictions vs ground truth.
    
    Args:
        image: Input image (numpy array, RGB)
        pred_heatmap: Predicted heatmap (num_classes, H, W)
        gt_heatmap: Ground truth heatmap (num_classes, H, W)
        threshold: Detection threshold
        class_names: List of class names
        save_path: Path to save visualization
    """
    if isinstance(pred_heatmap, torch.Tensor):
        pred_heatmap = pred_heatmap.cpu().numpy()
    if isinstance(gt_heatmap, torch.Tensor):
        gt_heatmap = gt_heatmap.cpu().numpy()
    
    num_classes = pred_heatmap.shape[0]
    img_h, img_w = image.shape[:2]
    h, w = pred_heatmap.shape[1], pred_heatmap.shape[2]
    
    # Extract centroids
    pred_centroids = get_centroids_from_heatmap(pred_heatmap, threshold)
    gt_centroids = get_centroids_from_heatmap(gt_heatmap, threshold=0.5)
    
    # Match centroids
    matches, unmatched_preds, unmatched_gts = match_centroids(pred_centroids, gt_centroids)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Ground truth
    gt_vis = image.copy()
    for class_id, cx, cy in gt_centroids:
        x = int(cx * img_w)
        y = int(cy * img_h)
        color = (0, 255, 0)  # Green
        cv2.circle(gt_vis, (x, y), 5, color, -1)
        if class_names:
            cv2.putText(gt_vis, class_names[class_id], (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    axes[0].imshow(cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Ground Truth ({len(gt_centroids)} objects)')
    axes[0].axis('off')
    
    # 2. Predictions
    pred_vis = image.copy()
    for class_id, cx, cy in pred_centroids:
        x = int(cx * img_w)
        y = int(cy * img_h)
        color = (255, 0, 0)  # Red
        cv2.circle(pred_vis, (x, y), 5, color, -1)
        if class_names:
            cv2.putText(pred_vis, class_names[class_id], (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    axes[1].imshow(cv2.cvtColor(pred_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Predictions ({len(pred_centroids)} objects)')
    axes[1].axis('off')
    
    # 3. Comparison (TP=green, FP=red, FN=yellow)
    comp_vis = image.copy()
    
    # Draw matches (TP) in green
    for pred_idx, gt_idx in matches:
        _, cx, cy = pred_centroids[pred_idx]
        x = int(cx * img_w)
        y = int(cy * img_h)
        cv2.circle(comp_vis, (x, y), 5, (0, 255, 0), -1)  # Green
    
    # Draw false positives in red
    for pred_idx in unmatched_preds:
        class_id, cx, cy = pred_centroids[pred_idx]
        x = int(cx * img_w)
        y = int(cy * img_h)
        cv2.circle(comp_vis, (x, y), 5, (255, 0, 0), -1)  # Red
    
    # Draw false negatives in yellow
    for gt_idx in unmatched_gts:
        class_id, cx, cy = gt_centroids[gt_idx]
        x = int(cx * img_w)
        y = int(cy * img_h)
        cv2.circle(comp_vis, (x, y), 5, (255, 255, 0), -1)  # Yellow
    
    axes[2].imshow(cv2.cvtColor(comp_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Comparison (TP={len(matches)}, FP={len(unmatched_preds)}, FN={len(unmatched_gts)})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    """
    Test evaluation functions.
    """
    print("=" * 60)
    print("FOMO Evaluation Utilities Test")
    print("=" * 60)
    
    # Test 1: Centroid extraction
    print("\n[Test 1] Centroid Extraction")
    print("-" * 60)
    
    # Create dummy heatmap
    heatmap = torch.zeros(2, 20, 20)
    heatmap[0, 10, 10] = 0.9  # Class 0 object
    heatmap[1, 5, 15] = 0.8   # Class 1 object
    heatmap[0, 15, 5] = 0.7   # Another class 0 object
    
    centroids = get_centroids_from_heatmap(heatmap, threshold=0.5)
    print(f"Extracted {len(centroids)} centroids:")
    for class_id, cx, cy in centroids:
        print(f"  Class {class_id}: ({cx:.3f}, {cy:.3f})")
    
    assert len(centroids) == 3, f"Expected 3 centroids, got {len(centroids)}"
    print("✓ Test passed")
    
    # Test 2: Centroid matching
    print("\n[Test 2] Centroid Matching")
    print("-" * 60)
    
    pred_centroids = [
        (0, 0.50, 0.50),  # Should match gt[0]
        (1, 0.30, 0.30),  # Should match gt[1]
        (0, 0.80, 0.80),  # False positive
    ]
    
    gt_centroids = [
        (0, 0.51, 0.49),  # Should match pred[0]
        (1, 0.31, 0.29),  # Should match pred[1]
        (0, 0.10, 0.10),  # False negative
    ]
    
    matches, unmatched_preds, unmatched_gts = match_centroids(
        pred_centroids, gt_centroids, distance_threshold=0.05
    )
    
    print(f"Matches: {len(matches)}")
    print(f"Unmatched predictions: {len(unmatched_preds)}")
    print(f"Unmatched ground truth: {len(unmatched_gts)}")
    
    assert len(matches) == 2, "Expected 2 matches"
    assert len(unmatched_preds) == 1, "Expected 1 FP"
    assert len(unmatched_gts) == 1, "Expected 1 FN"
    print("✓ Test passed")
    
    # Test 3: Metrics computation
    print("\n[Test 3] Metrics Computation")
    print("-" * 60)
    
    metrics = evaluate_centroids(pred_centroids, gt_centroids, num_classes=2)
    
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1']:.3f}")
    
    assert abs(metrics['precision'] - 2/3) < 0.01, "Precision should be 2/3"
    assert abs(metrics['recall'] - 2/3) < 0.01, "Recall should be 2/3"
    print("✓ Test passed")
    
    print("\n" + "=" * 60)
    print("✓ All evaluation tests passed!")
    print("=" * 60)
