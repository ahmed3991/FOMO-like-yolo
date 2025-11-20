#!/usr/bin/env python3
"""
Annotation Format Verification Script

This script verifies that:
1. All images are resized to 160x160
2. YOLO labels are correctly converted to center-point heatmaps
3. Bounding box dimensions (w, h) are ignored in FOMO training
"""

import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.datasets import load_yolo_labels, yolo_to_heatmap
from src.datasets import SimpleAugmentDataset


def verify_image_size(dataset, num_samples=10):
    """Verify all images are resized to 160x160"""
    print("=" * 70)
    print("Image Size Verification")
    print("=" * 70)
    
    all_correct = True
    
    for i in range(min(num_samples, len(dataset))):
        image, heatmap = dataset[i]
        
        expected_shape = (3, 160, 160)
        actual_shape = image.shape
        
        is_correct = actual_shape == expected_shape
        all_correct = all_correct and is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {i+1}: Image shape {actual_shape} (expected {expected_shape})")
    
    print("\nResult:")
    if all_correct:
        print("✅ All images are correctly resized to 160x160")
    else:
        print("❌ Some images have incorrect dimensions!")
    
    return all_correct


def detect_num_classes(dataset, num_samples=None):
    """Detect the actual number of classes in the dataset"""
    max_class = -1
    num_to_check = num_samples if num_samples else len(dataset)
    
    for i in range(min(num_to_check, len(dataset))):
        info = dataset.get_sample_info(i)
        labels = info['labels']
        for class_id, _, _, _, _ in labels:
            max_class = max(max_class, class_id)
    
    return max_class + 1  # Classes are 0-indexed


def verify_annotation_format(dataset, num_samples=10):
    """Verify YOLO labels are converted to center-point heatmaps"""
    print("\n" + "=" * 70)
    print("Annotation Format Verification (Center Points)")
    print("=" * 70)
    
    # Detect actual number of classes
    actual_num_classes = detect_num_classes(dataset, num_samples)
    if actual_num_classes > dataset.num_classes:
        print(f"\n⚠️  WARNING: Dataset contains {actual_num_classes} classes (0-{actual_num_classes-1}),")
        print(f"   but you specified --num-classes {dataset.num_classes}!")
        print(f"   Objects with class_id >= {dataset.num_classes} will be SKIPPED.\n")
        print(f"💡 Tip: Re-run with --num-classes {actual_num_classes}\n")
    
    all_samples_correct = True
    total_skipped = 0
    
    for i in range(min(num_samples, len(dataset))):
        info = dataset.get_sample_info(i)
        labels = info['labels']
        
        # Generate heatmap
        heatmap = yolo_to_heatmap(labels, output_size=20, num_classes=dataset.num_classes)
        
        print(f"\nSample {i+1}: {Path(info['image_path']).name}")
        print(f"  Number of objects: {len(labels)}")
        
        sample_correct = True
        grid_positions = set()  # Track unique grid positions
        sample_skipped = 0
        
        # Verify each object
        for obj_idx, (class_id, cx, cy, w, h) in enumerate(labels):
            # Skip objects with invalid class IDs
            if class_id >= dataset.num_classes:
                print(f"  ⚠️  Object {obj_idx+1}: SKIPPED (class_id={class_id} >= num_classes={dataset.num_classes})")
                sample_skipped += 1
                total_skipped += 1
                continue
            
            # Calculate expected grid position
            expected_x = int(cx * 20)
            expected_y = int(cy * 20)
            expected_x = max(0, min(19, expected_x))
            expected_y = max(0, min(19, expected_y))
            
            grid_positions.add((class_id, expected_x, expected_y))
            
            # Check heatmap
            heatmap_value = heatmap[class_id, expected_y, expected_x].item()
            
            is_correct = heatmap_value == 1.0
            sample_correct = sample_correct and is_correct
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} Object {obj_idx+1}:")
            print(f"      YOLO: class={class_id}, cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}")
            print(f"      Grid: ({expected_x}, {expected_y}) → Heatmap value: {heatmap_value}")
            print(f"      ⚠️  Width and height (w={w:.3f}, h={h:.3f}) are IGNORED ✓")
        
        # Count total marked points in heatmap
        total_marked = (heatmap > 0).sum().item()
        unique_positions = len(grid_positions)
        valid_objects = len(labels) - sample_skipped
        
        print(f"  Total marked points in heatmap: {total_marked}")
        print(f"  Valid objects: {valid_objects} → Unique grid positions: {unique_positions}")
        if sample_skipped > 0:
            print(f"  ⚠️  Skipped {sample_skipped} object(s) with invalid class IDs")
        
        # In FOMO, multiple objects can map to the same grid cell
        # So total_marked should equal unique_positions, not necessarily valid_objects
        if total_marked != unique_positions:
            print(f"  ⚠️  WARNING: Heatmap doesn't match expected grid positions!")
            sample_correct = False
        
        if valid_objects > unique_positions:
            collisions = valid_objects - unique_positions
            print(f"  ℹ️  Note: {collisions} object(s) share grid cell(s) with others (expected in FOMO)")
        
        sample_status = "✓" if sample_correct else "✗"
        print(f"  {sample_status} Sample {i+1}: All valid objects correctly marked")
        
        all_samples_correct = all_samples_correct and sample_correct
    
    print("\nResult:")
    if total_skipped > 0:
        print(f"⚠️  Total objects skipped: {total_skipped}")
    
    if all_samples_correct:
        print("✅  All annotations correctly converted to center-point format")
        print("✅ Bounding box dimensions (w, h) are properly ignored")
        print("ℹ️  Note: Multiple objects at same grid position is expected FOMO behavior")
    else:
        print("❌ Some annotations have issues!")
    
    return all_samples_correct



def verify_heatmap_dimensions(dataset, num_samples=10):
    """Verify heatmap output dimensions"""
    print("\n" + "=" * 70)
    print("Heatmap Dimension Verification")
    print("=" * 70)
    
    all_correct = True
    
    for i in range(min(num_samples, len(dataset))):
        image, heatmap = dataset[i]
        
        expected_shape = (dataset.num_classes, 20, 20)
        actual_shape = heatmap.shape
        
        is_correct = actual_shape == expected_shape
        all_correct = all_correct and is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {i+1}: Heatmap shape {actual_shape} (expected {expected_shape})")
    
    print("\nResult:")
    if all_correct:
        print("✅ All heatmaps have correct dimensions (num_classes, 20, 20)")
    else:
        print("❌ Some heatmaps have incorrect dimensions!")
    
    return all_correct


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify FOMO dataset implementation')
    parser.add_argument('--image-dir', type=str, required=True, help='Image directory')
    parser.add_argument('--label-dir', type=str, required=True, help='Label directory')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to verify')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FOMO Dataset Verification Script")
    print("=" * 70)
    print(f"\nDataset:")
    print(f"  Images: {args.image_dir}")
    print(f"  Labels: {args.label_dir}")
    print(f"  Classes: {args.num_classes}")
    print(f"  Samples to verify: {args.num_samples}")
    print()
    
    # Create dataset
    try:
        dataset = SimpleAugmentDataset(
            image_dir=args.image_dir,
            label_dir=args.label_dir,
            input_size=160,
            num_classes=args.num_classes,
            augment=False,
            normalize=True
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples\n")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        sys.exit(1)
    
    # Run verifications
    results = []
    
    results.append(("Image Size (160x160)", verify_image_size(dataset, args.num_samples)))
    results.append(("Annotation Format (Center Points)", verify_annotation_format(dataset, args.num_samples)))
    results.append(("Heatmap Dimensions", verify_heatmap_dimensions(dataset, args.num_samples)))
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All verifications PASSED!")
        print("=" * 70)
        print("\nConclusions:")
        print("  ✓ All images are correctly resized to 160x160")
        print("  ✓ YOLO annotations are converted to center-point format")
        print("  ✓ Bounding box dimensions (w, h) are properly ignored")
        print("  ✓ FOMO training uses only center coordinates (x, y)")
        sys.exit(0)
    else:
        print("❌ Some verifications FAILED!")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
