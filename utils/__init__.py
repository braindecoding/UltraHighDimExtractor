"""
Utility modules for UltraHighDimWaveletExtractor
===============================================

This module contains utility functions for data validation, visualization,
and quality metrics.

Modules:
- validation: Data validation functions
- visualization: Feature and result visualization
- metrics: Quality metrics and evaluation
"""

from .validation import validate_eeg_data, validate_features, DataValidator
from .metrics import FeatureQualityMetrics, ReconstructionMetrics
# from .visualization import FeatureVisualizer, ResultsVisualizer  # TODO: Implement later

__all__ = [
    'validate_eeg_data',
    'validate_features',
    'DataValidator',
    'FeatureQualityMetrics',
    'ReconstructionMetrics',
    # 'FeatureVisualizer',
    # 'ResultsVisualizer'
]
