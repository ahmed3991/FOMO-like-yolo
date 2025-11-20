"""
TFLite Model Testing Script

Test the quantized TFLite model to verify it works correctly.

Usage:
    python test_tflite_model.py --model models/model_quantized.tflite --image test_image.jpg
"""

import argparse
import numpy as np
import cv2
from pathlib import Path


def test_tflite_model(model_path, image_path=None):
    """Test TFLite model with optional test image."""
    
    try:
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow not installed")
        print("Install with: pip install tensorflow")
        return False
    
    print("=" * 70)
    print("TFLite Model Test")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n✓ Model loaded successfully!")
    print(f"\nModel Details:")
    print(f"  Input:")
    print(f"    Shape: {input_details[0]['shape']}")
    print(f"    Dtype: {input_details[0]['dtype']}")
    print(f"    Quantization: scale={input_details[0]['quantization'][0]:.6f}, "
          f"zero_point={input_details[0]['quantization'][1]}")
    
    print(f"\n  Output:")
    print(f"    Shape: {output_details[0]['shape']}")
    print(f"    Dtype: {output_details[0]['dtype']}")
    print(f"    Quantization: scale={output_details[0]['quantization'][0]:.6f}, "
          f"zero_point={output_details[0]['quantization'][1]}")
    
    # Prepare input
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    if image_path and Path(image_path).exists():
        print(f"\n✓ Loading test image: {image_path}")
        
        # Load and preprocess image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  Error: Could not load image")
            return False
        
        h, w = input_shape[1:3]
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # Quantize if needed
        if input_dtype == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            img = (img / input_scale + input_zero_point)
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        test_input = img[np.newaxis, ...]
        
    else:
        print(f"\n✓ Creating random test input")
        
        if input_dtype == np.uint8:
            test_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
        else:
            test_input = np.random.rand(*input_shape).astype(np.float32)
    
    # Run inference
    print("\n✓ Running inference...")
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize if needed
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_float = (output.astype(np.float32) - output_zero_point) * output_scale
    else:
        output_float = output
    
    print(f"\n✓ Inference successful!")
    print(f"\nOutput Statistics:")
    print(f"  Shape: {output.shape}")
    print(f"  Raw output range: [{output.min()}, {output.max()}]")
    print(f"  Dequantized range: [{output_float.min():.4f}, {output_float.max():.4f}]")
    print(f"  Mean: {output_float.mean():.4f}")
    print(f"  Std:  {output_float.std():.4f}")
    
    # Apply sigmoid to get probabilities
    from scipy.special import expit
    probs = expit(output_float)
    
    print(f"\nProbabilities (after sigmoid):")
    print(f"  Range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  Mean:  {probs.mean():.4f}")
    
    # Count detections above threshold
    threshold = 0.5
    detections = (probs > threshold).sum()
    print(f"  Detections (>{threshold}): {detections}")
    
    # Model size
    model_size_kb = Path(model_path).stat().st_size / 1024
    print(f"\nModel Size: {model_size_kb:.2f} KB")
    
    if model_size_kb < 500:
        print(f"  ✓ Fits ESP32 constraint (<500 KB)")
    else:
        print(f"  ⚠️  May be too large for ESP32 (target: <500 KB)")
    
    print("\n" + "=" * 70)
    print("✓ Test Complete!")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test TFLite model')
    
    parser.add_argument('--model', type=str, default='models/model_quantized.tflite',
                       help='Path to TFLite model')
    parser.add_argument('--image', type=str,
                       help='Optional test image path')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("\nPlease export the model first:")
        print("  python export_and_quantize.py --checkpoint checkpoints_transfer/best_model_transfer.pth --all")
        return
    
    test_tflite_model(args.model, args.image)


if __name__ == "__main__":
    main()
