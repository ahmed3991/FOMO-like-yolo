"""
Verification script for FOMO-like YOLOv8 Micro Model

Tests the model with different input sizes and verifies:
- Output shapes
- Parameter count
- Forward pass completion
"""

import torch
from src.models import MicroYOLO


def verify_model():
    """Verify the MicroYOLO model implementation."""
    
    print("=" * 60)
    print("FOMO-like YOLOv8 Micro Model Verification")
    print("=" * 60)
    
    # Test with 160x160 input
    print("\n[Test 1] 160x160 Input")
    print("-" * 60)
    model_160 = MicroYOLO(nc=2, input_size=160)
    x_160 = torch.randn(1, 3, 160, 160)
    
    with torch.no_grad():
        y_160 = model_160(x_160)
    
    print(f"Input shape:    {tuple(x_160.shape)}")
    print(f"Output shape:   {tuple(y_160.shape)}")
    print(f"Expected shape: (1, 2, 20, 20)")
    
    assert y_160.shape == (1, 2, 20, 20), f"Expected (1, 2, 20, 20), got {y_160.shape}"
    assert y_160.min() >= 0.0 and y_160.max() <= 1.0, "Output should be in [0, 1] due to sigmoid"
    print("✓ Shape and activation validation passed!")
    
    # Test with 96x96 input
    print("\n[Test 2] 96x96 Input")
    print("-" * 60)
    model_96 = MicroYOLO(nc=2, input_size=96)
    x_96 = torch.randn(1, 3, 96, 96)
    
    with torch.no_grad():
        y_96 = model_96(x_96)
    
    print(f"Input shape:    {tuple(x_96.shape)}")
    print(f"Output shape:   {tuple(y_96.shape)}")
    print(f"Expected shape: (1, 2, 12, 12)")
    
    assert y_96.shape == (1, 2, 12, 12), f"Expected (1, 2, 12, 12), got {y_96.shape}"
    assert y_96.min() >= 0.0 and y_96.max() <= 1.0, "Output should be in [0, 1] due to sigmoid"
    print("✓ Shape and activation validation passed!")
    
    # Test with multiple classes
    print("\n[Test 3] Multiple Classes (nc=5)")
    print("-" * 60)
    model_multi = MicroYOLO(nc=5, input_size=160)
    x_multi = torch.randn(1, 3, 160, 160)
    
    with torch.no_grad():
        y_multi = model_multi(x_multi)
    
    print(f"Input shape:    {tuple(x_multi.shape)}")
    print(f"Output shape:   {tuple(y_multi.shape)}")
    print(f"Expected shape: (1, 5, 20, 20)")
    
    assert y_multi.shape == (1, 5, 20, 20), f"Expected (1, 5, 20, 20), got {y_multi.shape}"
    print("✓ Multi-class validation passed!")
    
    # Test batch processing
    print("\n[Test 4] Batch Processing")
    print("-" * 60)
    batch_size = 4
    x_batch = torch.randn(batch_size, 3, 160, 160)
    
    with torch.no_grad():
        y_batch = model_160(x_batch)
    
    print(f"Input shape:    {tuple(x_batch.shape)}")
    print(f"Output shape:   {tuple(y_batch.shape)}")
    print(f"Expected shape: ({batch_size}, 2, 20, 20)")
    
    assert y_batch.shape == (batch_size, 2, 20, 20), f"Expected ({batch_size}, 2, 20, 20), got {y_batch.shape}"
    print("✓ Batch processing validation passed!")
    
    # Parameter count
    print("\n[Model Statistics]")
    print("-" * 60)
    total_params = sum(p.numel() for p in model_160.parameters())
    trainable_params = sum(p.numel() for p in model_160.parameters() if p.requires_grad)
    
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (approx):  {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # Architecture summary
    print("\n[Architecture Summary]")
    print("-" * 60)
    print("Backbone:")
    print("  - Stem:   Conv(3→16, stride=2)")
    print("  - Stage1: C2f(16→32)")
    print("  - Stage2: C2f(32→64, stride=2)")
    print("  - Stage3: C2f(64→128, stride=2)")
    print("  - Stage4: C2f(128→160)")
    print("\nHead:")
    print("  - Head1:  Conv(160→32)")
    print("  - Head2:  Conv(32→nc)")
    print("  - Output: Sigmoid activation")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    verify_model()
