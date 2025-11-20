"""
Download module for datasets.

Provides utilities for downloading and preparing various datasets for FOMO training.
"""

from .coco import download_coco128, download_roboflow_dataset
from .kaggle import download_kaggle_grocery, setup_kaggle_credentials, extract_and_prepare_grocery
from .utils import (
    download_file,
    extract_zip,
    create_train_val_split,
    verify_dataset,
    update_config_paths
)

__all__ = [
    'download_coco128',
    'download_roboflow_dataset',
    'download_kaggle_grocery',
    'setup_kaggle_credentials',
    'extract_and_prepare_grocery',
    'download_file',
    'extract_zip',
    'create_train_val_split',
    'verify_dataset',
    'update_config_paths',
]
