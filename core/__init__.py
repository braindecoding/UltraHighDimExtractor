"""
Core modules for UltraHighDimWaveletExtractor
============================================

This module contains the core functionality for ultra-high dimensional
wavelet feature extraction from EEG data.

Modules:
- base: Base classes and interfaces
- preprocessing: EEG preprocessing pipeline
- dwt_extractor: Discrete Wavelet Transform feature extraction
- wpd_extractor: Wavelet Packet Decomposition feature extraction
- ultra_extractor: Main ultra-high dimensional extractor
- pipeline: Complete image reconstruction pipeline
"""

# Import only working modules for now
try:
    from .base import WaveletFeatureBase, FeatureExtractorInterface, WaveletAnalyzer
    from .preprocessing import EEGPreprocessor, create_optimal_preprocessor

    # These will be imported later after fixing dependencies
    # from .dwt_extractor import DWTExtractor
    # from .wpd_extractor import WPDExtractor
    # from .ultra_extractor import UltraHighDimExtractor
    # from .pipeline import ImageReconstructionPipeline

    print("✅ Core base modules loaded successfully")

except ImportError as e:
    print(f"⚠️ Warning: Core imports failed: {e}")

__all__ = [
    'WaveletFeatureBase',
    'FeatureExtractorInterface',
    'WaveletAnalyzer',
    'EEGPreprocessor',
    'create_optimal_preprocessor',
    # 'DWTExtractor',
    # 'WPDExtractor',
    # 'UltraHighDimExtractor',
    # 'ImageReconstructionPipeline'
]
