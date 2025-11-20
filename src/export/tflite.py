"""
Model Export and Quantization for ESP32 Deployment

This script exports the trained PyTorch model to ONNX and TFLite formats,
applies INT8 quantization, and prepares it for ESP32 deployment.

Usage:
    # Export to ONNX
    python export_and_quantize.py --checkpoint checkpoints_transfer/best_model_transfer.pth --export-onnx
    
    # Export to TFLite with quantization
    python export_and_quantize.py --checkpoint checkpoints_transfer/best_model_transfer.pth --export-tflite --quantize
    
    # Complete workflow
    python export_and_quantize.py --checkpoint checkpoints_transfer/best_model_transfer.pth --all
"""

import os
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.onnx

from src.models import MicroYOLO


def export_to_onnx(checkpoint_path, output_path="models/model.onnx"):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        output_path: Output path for ONNX model
    """
    print("=" * 70)
    print("Exporting to ONNX")
    print("=" * 70)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    num_classes = checkpoint['config']['num_classes']
    input_size = checkpoint['config']['input_size']
    
    print(f"\nModel info:")
    print(f"  Classes: {num_classes}")
    print(f"  Input size: {input_size}×{input_size}")
    print(f"  Output size: {input_size//8}×{input_size//8}")
    
    # Create model
    model = MicroYOLO(nc=num_classes, input_size=input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting to {output_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ ONNX model saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
    
    return str(output_path)


def onnx_to_tensorflow(onnx_path, output_dir="models/tf_model"):
    """Convert ONNX model to TensorFlow SavedModel format."""
    print("\n" + "=" * 70)
    print("Converting ONNX to TensorFlow")
    print("=" * 70)
    
    try:
        import onnx
        from onnx_tf.backend import prepare
    except ImportError:
        print("\n✗ Error: onnx-tf not installed")
        print("  Install with: pip install onnx onnx-tf")
        return None
    
    # Load ONNX model
    print(f"\nLoading ONNX model from {onnx_path}...")
    onnx_model = onnx.load(onnx_path)
    
    # Convert to TensorFlow
    print("Converting to TensorFlow...")
    tf_rep = prepare(onnx_model)
    
    # Save as SavedModel
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving TensorFlow model to {output_path}...")
    tf_rep.export_graph(str(output_path))
    
    print(f"✓ TensorFlow model saved to: {output_path}")
    
    return str(output_path)


def tensorflow_to_tflite(tf_model_path, output_path="models/model.tflite", quantize=False, 
                         representative_dataset=None):
    """
    Convert TensorFlow SavedModel to TFLite format with optional quantization.
    
    Args:
        tf_model_path: Path to TensorFlow SavedModel
        output_path: Output path for TFLite model
        quantize: Whether to apply INT8 quantization
        representative_dataset: Generator for representative dataset (required for quantization)
    """
    print("\n" + "=" * 70)
    print(f"Converting to TFLite" + (" with INT8 Quantization" if quantize else ""))
    print("=" * 70)
    
    try:
        import tensorflow as tf
    except ImportError:
        print("\n✗ Error: tensorflow not installed")
        print("  Install with: pip install tensorflow")
        return None
    
    # Load TensorFlow model
    print(f"\nLoading TensorFlow model from {tf_model_path}...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    if quantize:
        print("\nApplying INT8 quantization...")
        
        # Set optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set representative dataset if provided
        if representative_dataset:
            converter.representative_dataset = representative_dataset
        
        # Ensure INT8 quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        print("  Quantization settings:")
        print("    - Full integer quantization (INT8)")
        print("    - Input/Output: UINT8")
    else:
        print("\nNo quantization applied (FP32)")
    
    # Convert
    print("\nConverting...")
    tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"\n✓ TFLite model saved to: {output_path}")
    print(f"  File size: {size_kb:.2f} KB")
    
    if size_kb < 500:
        print(f"  ✓ Model fits ESP32 constraint (<500 KB)")
    else:
        print(f"  ⚠️  Model may be too large for ESP32 (target: <500 KB)")
    
    return str(output_path)


def create_representative_dataset(input_size=160, num_samples=100):
    """
    Create a representative dataset for quantization.
    
    This generates random samples. For better accuracy, use actual validation images.
    """
    def representative_data_gen():
        for _ in range(num_samples):
            # Generate random input (normalized)
            data = np.random.rand(1, input_size, input_size, 3).astype(np.float32)
            # Normalize like validation data
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            data = (data - mean) / std
            yield [data]
    
    return representative_data_gen


def pytorch_to_tflite_direct(checkpoint_path, output_path="models/model_quantized.tflite", 
                            quantize=True):
    """
    Direct conversion from PyTorch to TFLite (bypassing ONNX/TF).
    
    This is a simplified approach using torch.jit and ai_edge_torch if available.
    """
    print("=" * 70)
    print("Direct PyTorch to TFLite Conversion")
    print("=" * 70)
    
    # Note: This would require ai_edge_torch or similar
    # For now, we'll use the ONNX -> TF -> TFLite pipeline
    
    print("\n⚠️  Direct conversion not implemented yet")
    print("   Using ONNX -> TF -> TFLite pipeline instead")
    
    return None


def verify_tflite_model(tflite_path, input_size=160):
    """Verify TFLite model by running inference."""
    print("\n" + "=" * 70)
    print("Verifying TFLite Model")
    print("=" * 70)
    
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not available for verification")
        return
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nModel details:")
    print(f"  Input shape:  {input_details[0]['shape']}")
    print(f"  Input dtype:  {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")
    
    # Test with random input
    print("\nRunning test inference...")
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.uint8:
        test_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    else:
        test_input = np.random.rand(*input_shape).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✓ Inference successful!")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description='Export and quantize model for ESP32')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--export-onnx', action='store_true',
                       help='Export to ONNX format')
    parser.add_argument('--export-tflite', action='store_true',
                       help='Export to TFLite format')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply INT8 quantization to TFLite')
    parser.add_argument('--all', action='store_true',
                       help='Run complete export pipeline (ONNX -> TF -> TFLite quantized)')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for exported models')
    parser.add_argument('--verify', action='store_true',
                       help='Verify exported models')
    
    args = parser.parse_args()
    
    if not any([args.export_onnx, args.export_tflite, args.all]):
        parser.print_help()
        print("\nExample usage:")
        print("  # Complete pipeline with quantization")
        print("  python export_and_quantize.py --checkpoint checkpoints_transfer/best_model_transfer.pth --all")
        print("\n  # Just export to ONNX")
        print("  python export_and_quantize.py --checkpoint checkpoints_transfer/best_model_transfer.pth --export-onnx")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model info
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    input_size = checkpoint['config']['input_size']
    
    onnx_path = None
    tf_path = None
    tflite_path = None
    
    # Export pipeline
    if args.export_onnx or args.all:
        onnx_path = export_to_onnx(
            args.checkpoint,
            output_path=str(output_dir / "model.onnx")
        )
    
    if args.export_tflite or args.all:
        # Need ONNX first
        if onnx_path is None:
            onnx_path = str(output_dir / "model.onnx")
            if not Path(onnx_path).exists():
                print("\n⚠️  ONNX model not found, exporting first...")
                onnx_path = export_to_onnx(args.checkpoint, onnx_path)
        
        # Convert ONNX to TF
        tf_path = onnx_to_tensorflow(
            onnx_path,
            output_dir=str(output_dir / "tf_model")
        )
        
        if tf_path:
            # Convert TF to TFLite
            rep_dataset = None
            if args.quantize or args.all:
                print("\nCreating representative dataset for quantization...")
                rep_dataset = create_representative_dataset(input_size, num_samples=100)
            
            tflite_path = tensorflow_to_tflite(
                tf_path,
                output_path=str(output_dir / ("model_quantized.tflite" if (args.quantize or args.all) else "model.tflite")),
                quantize=(args.quantize or args.all),
                representative_dataset=rep_dataset
            )
    
    # Verify
    if args.verify and tflite_path:
        verify_tflite_model(tflite_path, input_size)
    
    # Summary
    print("\n" + "=" * 70)
    print("Export Complete!")
    print("=" * 70)
    print(f"\nExported models in: {output_dir}/")
    if onnx_path:
        print(f"  - ONNX:   {Path(onnx_path).name}")
    if tf_path:
        print(f"  - TF:     {Path(tf_path).name}/")
    if tflite_path:
        print(f"  - TFLite: {Path(tflite_path).name}")
    
    print("\nNext steps:")
    print("1. Test TFLite model: python test_tflite_model.py")
    print("2. Deploy to ESP32: See esp32_deployment_guide.md")


if __name__ == "__main__":
    main()
