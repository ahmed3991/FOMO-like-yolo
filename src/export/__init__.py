"""
Export module for FOMO model.

Provides utilities for exporting trained models to various formats:
- ONNX format
- TFLite format (with optional quantization)
- C header files for embedded deployment
"""

from .onnx import export_to_onnx
from .tflite import (
    onnx_to_tensorflow,
    tensorflow_to_tflite,
    create_representative_dataset,
    verify_tflite_model
)
from .c_header import model_to_c_header

__all__ = [
    'export_to_onnx',
    'onnx_to_tensorflow',
    'tensorflow_to_tflite',
    'create_representative_dataset',
    'verify_tflite_model',
    'model_to_c_header',
]
