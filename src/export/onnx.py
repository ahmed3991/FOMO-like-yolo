"""
Simplified ONNX Export for ESP32

This script exports only to ONNX format, which can be converted to TFLite
using other tools or deployed directly with ONNX Runtime.

For ESP32, you have two options:
1. Use ONNX Runtime Micro (newer, better)
2. Manual conversion: ONNX → TFLite using external tools

Usage:
    python export_onnx_only.py --checkpoint checkpoints_transfer/best_model_transfer.pth
"""

import argparse
from pathlib import Path
import torch
from src.models import MicroYOLO


def export_to_onnx(checkpoint_path, output_path="models/model.onnx", opset_version=11):
    """Export PyTorch model to ONNX format."""
    
    print("=" * 70)
    print("Exporting to ONNX for ESP32")
    print("=" * 70)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    num_classes = checkpoint['config']['num_classes']
    input_size = checkpoint['config']['input_size']
    
    print(f"\nModel Configuration:")
    print(f"  Classes: {num_classes}")
    print(f"  Input size: {input_size}×{input_size}")
    print(f"  Output size: {input_size//8}×{input_size//8}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    print(f"  Epoch: {checkpoint['epoch']}")
    
    # Create and load model
    model = MicroYOLO(nc=num_classes, input_size=input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\n✓ Model forward pass successful")
    print(f"  Output shape: {output.shape}")
    
    # Prepare output path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    print(f"\nExporting to ONNX...")
    print(f"  Output: {output_path}")
    print(f"  Opset version: {opset_version}")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    file_size_kb = output_path.stat().st_size / 1024
    
    print(f"\n✓ ONNX export successful!")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB ({file_size_kb:.1f} KB)")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verification passed")
    except ImportError:
        print(f"⚠️  ONNX verification skipped (onnx not installed)")
    except Exception as e:
        print(f"⚠️  ONNX verification failed: {e}")
    
    return str(output_path), file_size_kb


def print_next_steps(onnx_path, model_size_kb):
    """Print instructions for next steps."""
    
    print("\n" + "=" * 70)
    print("Next Steps for ESP32 Deployment")
    print("=" * 70)
    
    print(f"\n**ONNX Model Ready**: {onnx_path}")
    print(f"  Size: {model_size_kb:.1f} KB")
    
    print("\n**Option 1: ONNX Runtime Micro (Recommended for ESP32)**")
    print("  Modern alternative to TFLite, better support for newer models")
    print("  Steps:")
    print("  1. Visit: https://github.com/microsoft/onnxruntime")
    print("  2. Use ONNX Runtime for ESP32-S3")
    print("  3. Deploy ONNX model directly (no conversion needed)")
    
    print("\n**Option 2: Convert to TFLite (External Tools)**")
    print("  If you must use TFLite, use these tools:")
    print()
    print("  A. Using Online Converter:")
    print("     Visit: https://netron.app to visualize model")
    print("     Use: https://convertmodel.com or similar services")
    print()
    print("  B. Using Docker (TensorFlow)")
    print("     docker run -it tensorflow/tensorflow:latest-gpu bash")
    print("     # Inside container, install onnx-tf and convert")
    print()
    print("  C. Using tf2onnx in reverse (manual)")
    print("     See: esp32_deployment_guide.md for detailed steps")
    
    print("\n**Option 3: Direct PyTorch Mobile**")
    print("  Use PyTorch Mobile for ESP32:")
    print("  1. Export to TorchScript:")
    print("     model_scripted = torch.jit.script(model)")
    print("     model_scripted.save('model.pt')")
    print("  2. Use PyTorch Mobile runtime on ESP32")
    
    print("\n**Quantization (To reduce size further)**")
    print(f"  Current: {model_size_kb:.1f} KB (FP32)")
    print(f"  Target: <500 KB for ESP32")
    print()
    if model_size_kb > 500:
        print("  ⚠️  Model is larger than ESP32 target")
        print("  Recommendations:")
        print("  - Try 96×96 input (reduce from 160×160)")
        print("  - Reduce number of classes")
        print("  - Use model pruning techniques")
    else:
        print("  ✓ Model already under 500 KB!")
    
    print("\n**Testing the ONNX Model**")
    print("  python test_onnx_model.py")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX for ESP32')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--output', type=str, default='models/model.onnx',
                       help='Output ONNX file path')
    parser.add_argument('--opset-version', type=int, default=11,
                       help='ONNX opset version (11 is widely supported)')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Export to ONNX
    onnx_path, size_kb = export_to_onnx(
        args.checkpoint,
        args.output,
        args.opset_version
    )
    
    # Print next steps
    print_next_steps(onnx_path, size_kb)


if __name__ == "__main__":
    main()
