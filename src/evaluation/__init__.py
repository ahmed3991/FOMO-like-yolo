"""
Evaluation module for FOMO model.

Provides utilities for:
- Computing evaluation metrics (precision, recall, F1)
- Extracting centroids from heatmaps
- Visualizing predictions
- Verifying models and data
- Testing TFLite models
"""

from .metrics import (
    get_centroids_from_heatmap,
    match_centroids,
    evaluate_centroids,
    visualize_predictions
)

__all__ = [
    'get_centroids_from_heatmap',
    'match_centroids',
    'evaluate_centroids',
    'visualize_predictions',
]
