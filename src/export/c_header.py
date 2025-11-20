"""
Simple converter: Any binary model file to C header

This works with ONNX, TFLite, or any binary model format.
For ESP32 deployment.

Usage:
    # From ONNX directly
    python model_to_c_header.py --model models/model.onnx --output models/model_onnx.h
    
    # From TFLite (if you have it)
    python model_to_c_header.py --model models/model.tflite --output models/model.h
"""

import argparse
from pathlib import Path


def model_to_c_header(model_path, header_path, var_name=None):
    """
    Convert any binary model file to C header.
    
    Args:
        model_path: Path to model file (.onnx, .tflite, etc.)
        header_path: Output C header file path
        var_name: Variable name (defaults to filename)
    """
    model_path = Path(model_path)
    header_path = Path(header_path)
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return False
    
    # Default variable name from filename
    if var_name is None:
        var_name = model_path.stem.replace('-', '_').replace('.', '_') + "_data"
    
    # Read model file
    with open(model_path, 'rb') as f:
        model_data = f.read()
    
    model_size = len(model_data)
    model_size_kb = model_size / 1024
    
    print("=" * 70)
    print("Model to C Header Converter")
    print("=" * 70)
    print(f"\nInput:  {model_path}")
    print(f"Output: {header_path}")
    print(f"Size:   {model_size:,} bytes ({model_size_kb:.2f} KB)")
    
    if model_size_kb > 500:
        print(f"⚠️  Warning: Model size ({model_size_kb:.1f} KB) exceeds ESP32 target (<500 KB)")
    else:
        print(f"✓ Model size OK for ESP32")
    
    # Generate C header
    guard_name = header_path.stem.upper() + "_H"
    
    header_content = f"""/* Auto-generated model header file */
/* Source: {model_path.name} */
/* Size: {model_size:,} bytes ({model_size_kb:.2f} KB) */
/* Generated for ESP32 deployment */

#ifndef {guard_name}
#define {guard_name}

#ifdef __cplusplus
extern "C" {{
#endif

/* Align model data for optimal memory access */
#ifdef __has_attribute
#define ALIGN(X) __attribute__((aligned(X)))
#else
#define ALIGN(X)
#endif

/* Model data array */
ALIGN(16) const unsigned char {var_name}[] = {{
"""
    
    # Add model bytes (12 per line)
    bytes_per_line = 12
    for i in range(0, len(model_data), bytes_per_line):
        chunk = model_data[i:i+bytes_per_line]
        hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
        header_content += f"    {hex_values},\n"
    
    # Remove trailing comma and close array
    header_content = header_content.rstrip(',\n') + '\n};\n\n'
    
    header_content += f"""/* Model size constant */
const unsigned int {var_name}_len = {model_size};

#ifdef __cplusplus
}}
#endif

#endif /* {guard_name} */
"""
    
    # Write header file
    header_path.parent.mkdir(parents=True, exist_ok=True)
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    header_size_kb = header_path.stat().st_size / 1024
    
    print(f"\n✓ C header created: {header_path}")
    print(f"  Header size: {header_size_kb:.2f} KB")
    print(f"  Variable: {var_name}[{model_size}]")
    print(f"  Length: {var_name}_len")
    
    print("\n" + "=" * 70)
    print("Usage in ESP32 Code")
    print("=" * 70)
    print(f"""
#include "{header_path.name}"

// Access model data:
const unsigned char* model = {var_name};
unsigned int model_size = {var_name}_len;

// Example with TFLite Micro:
const tflite::Model* tflite_model = 
    tflite::GetModel({var_name});

// Or with ONNX Runtime:
// Load model from {var_name} array
""")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert binary model file to C header for ESP32'
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Input model file (ONNX, TFLite, etc.)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output C header file path')
    parser.add_argument('--var-name', type=str,
                       help='Variable name (default: derived from filename)')
    
    args = parser.parse_args()
    
    success = model_to_c_header(args.model, args.output, args.var_name)
    
    if success:
        print("\n✓ Conversion successful!")
        print("\nNext steps:")
        print("1. Copy the .h file to your ESP32 project")
        print("2. Include it in your main code")
        print("3. Load with your inference runtime (TFLite Micro or ONNX Runtime)")
    else:
        print("\n✗ Conversion failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
