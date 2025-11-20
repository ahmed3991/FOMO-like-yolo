#!/usr/bin/env python3
"""
Transfer Learning Script - CLI Wrapper

Convenient command-line interface for transfer learning on new datasets.

Usage:
    python scripts/transfer_learning.py \
        --checkpoint checkpoints/best_model.pth \
        --train-image-dir data/grocery/train/images \
        --train-label-dir data/grocery/train/labels \
        --val-image-dir data/grocery/valid/images \
        --val-label-dir data/grocery/valid/labels \
        --num-classes 43 \
        --epochs 20
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from src.training.transfer_learning import transfer_learn
    transfer_learn()
