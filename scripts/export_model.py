"""
Model Export CLI Script

Convenient command-line interface for exporting FOMO models to various formats.

Usage:
    # Export to all formats with quantization
    python scripts/export_model.py --checkpoint checkpoints/best_model.pth --all
    
    # Export only to ONNX
    python scripts/export_model.py --checkpoint checkpoints/best_model.pth --format onnx
    
    # Export to TFLite with quantization
    python scripts/export_model.py --checkpoint checkpoints/best_model.pth --format tflite --quantize
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.export import (
    export_to_onnx,
    onnx_to_tensorflow,
    tensorflow_to_tflite,
    create_representative_dataset,
    verify_tflite_model,
    model_to_c_header
)


def main():
    parser = argparse.ArgumentParser(description='Export FOMO model to various formats')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--format', type=str, choices=['onnx', 'tflite', 'all'],
                       default='all', help='Export format (default: all)')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply INT8 quantization to TFLite')
    parser.add_argument('--c-header', action='store_true',
                       help='Generate C header file from TFLite model')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for exported models')
    parser.add_argument('--verify', action='store_true',
                       help='Verify exported TFLite model')
    parser.add_argument('--all', action='store_true',
                       help='Export to all formats with quantization and C header')
    
    args = parser.parse_args()
    
    # If --all flag is set, override other settings
    if args.all:
        args.format = 'all'
        args.quantize = True
        args.c_header = True
        args.verify = True
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model info
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    input_size = checkpoint['config']['input_size']
    
    onnx_path = None
    tf_path = None
    tflite_path = None
    
    # Export to ONNX
    if args.format in ['onnx', 'all']:
        onnx_path = export_to_onnx(
            args.checkpoint,
            output_path=str(output_dir / "model.onnx")
        )
    
    # Export to TFLite
    if args.format in ['tflite', 'all']:
        # Need ONNX first
        if onnx_path is None:
            onnx_path = str(output_dir / "model.onnx")
            if not Path(onnx_path).exists():
                print("\n⚠️  ONNX model not found, exporting first...")
                onnx_path = export_to_onnx(args.checkpoint, onnx_path)
        
        # Convert ONNX to TensorFlow
        tf_path = onnx_to_tensorflow(
            onnx_path,
            output_dir=str(output_dir / "tf_model")
        )
        
        if tf_path:
            # Convert TensorFlow to TFLite
            rep_dataset = None
            if args.quantize:
                print("\nCreating representative dataset for quantization...")
                rep_dataset = create_representative_dataset(input_size, num_samples=100)
            
            tflite_path = tensorflow_to_tflite(
                tf_path,
                output_path=str(output_dir / ("model_quantized.tflite" if args.quantize else "model.tflite")),
                quantize=args.quantize,
                representative_dataset=rep_dataset
            )
    
    # Generate C header
    if args.c_header and tflite_path:
        model_to_c_header(
            tflite_path,
            header_path=str(output_dir / "model.h"),
            var_name="fomo_model_data"
        )
    
    # Verify TFLite model
    if args.verify and tflite_path:
        verify_tflite_model(tflite_path, input_size)
    
    # Summary
    print("\n" + "=" * 70)
    print("Export Complete!")
    print("=" * 70)
    print(f"\nExported models in: {output_dir}/")
    if onnx_path:
        print(f"  ✓ ONNX:   {Path(onnx_path).name}")
    if tflite_path:
        print(f"  ✓ TFLite: {Path(tflite_path).name}")
        if args.c_header:
            print(f"  ✓ C Header: model.h")
    
    print("\nNext steps:")
    print("  - Test TFLite: python -m src.evaluation.test_tflite --model", tflite_path if tflite_path else "models/model.tflite")
    print("  - Deploy to ESP32: Include model.h in your ESP32 project")


if __name__ == "__main__":
    main()
