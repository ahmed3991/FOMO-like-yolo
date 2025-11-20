#!/usr/bin/env python3
"""
Training Script - CLI Wrapper

Convenient command-line interface for training FOMO model from scratch.

Usage:
    python scripts/train.py --epochs 30 --batch-size 16
    python scripts/train.py --help
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from src.training.train import main
    main()
