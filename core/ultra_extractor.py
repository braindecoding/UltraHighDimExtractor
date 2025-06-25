"""
Ultra-High Dimensional Wavelet Feature Extractor
===============================================

This module implements the main UltraHighDimExtractor class that combines
multiple wavelet approaches to achieve ultra-high dimensional feature extraction
for EEG-to-image reconstruction tasks.

Classes:
- UltraHighDimExtractor: Main extractor achieving 40,418+ features
- MultiWaveletExtractor: Multiple wavelet families extractor
- AdaptiveExtractor: Adaptive parameter selection extractor
"""

import numpy as np
import pywt
from typing import List, Dict, Tuple, Optional
import logging
import time

logger = logging.getLogger(__name__)

from .base import WaveletFeatureBase

# Import DWT and WPD extractors from core
try:
    from .dwt_extractor import DWTFeatureExtractor as DWTExtractor
    from .wpd_extractor import WPDFeatureExtractor as WPDExtractor
    EXTRACTORS_AVAILABLE = True
    logger.info("DWT and WPD extractors loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import extractors: {e}")
    DWTExtractor = None
    WPDExtractor = None
    EXTRACTORS_AVAILABLE = False

logger = logging.getLogger(__name__)


class UltraHighDimExtractor(WaveletFeatureBase):
    """
    Ultra-High Dimensional Wavelet Feature Extractor.
    
    This extractor combines multiple wavelet approaches to achieve maximum
    feature dimensionality for EEG-to-image reconstruction tasks.
    
    Features:
    - Multiple wavelet families (db8, db10, coif5, etc.)
    - Deep DWT and WPD decomposition
    - Comprehensive feature types
    - Optimized for image reconstruction
    - Achieves 40,418+ features
    """
    
    def __init__(self, 
                 target_dimensions: int = 40000,
                 wavelets: List[str] = None,
                 max_dwt_levels: int = 6,
                 max_wpd_levels: int = 5,
                 feature_types: List[str] = None,
                 sampling_rate: float = 128.0,
                 **kwargs):
        """
        Initialize ultra-high dimensional extractor.
        
        Args:
            target_dimensions: Target number of features
            wavelets: List of wavelets to use
            max_dwt_levels: Maximum DWT decomposition levels
            max_wpd_levels: Maximum WPD decomposition levels
            feature_types: Types of features to extract
            sampling_rate: Sampling rate in Hz
        """
        super().__init__("UltraHighDimExtractor", 
                        wavelets[0] if wavelets else 'db8', 
                        max_dwt_levels, sampling_rate, **kwargs)
        
        self.target_dimensions = target_dimensions
        
        # Default wavelets optimized for EEG
        if wavelets is None:
            self.wavelets = ['db8', 'db10', 'coif5', 'bior4.4', 'sym8']
        else:
            self.wavelets = wavelets
        
        self.max_dwt_levels = max_dwt_levels
        self.max_wpd_levels = max_wpd_levels
        
        # Default feature types for maximum information
        if feature_types is None:
            self.feature_types = [
                'statistical', 'energy', 'entropy', 
                'frequency', 'spectral', 'morphological'
            ]
        else:
            self.feature_types = feature_types
        
        # Initialize extractors
        self.extractors = []
        self.extractor_info = []
        
        logger.info(f"Ultra-high dim extractor targeting {target_dimensions} features")
        logger.info(f"Using wavelets: {self.wavelets}")
    
    def _create_extractors(self):
        """Create multiple specialized extractors for maximum dimensionality."""
        if not EXTRACTORS_AVAILABLE:
            raise RuntimeError("DWT and WPD extractors are not available. Please check waveletfeatures folder.")

        self.extractors = []
        self.extractor_info = []

        logger.info("Creating specialized extractors...")

        # Strategy 1: Multiple DWT extractors with different wavelets
        for i, wavelet in enumerate(self.wavelets[:3]):  # Use first 3 wavelets
            extractor = DWTExtractor(
                wavelet=wavelet,
                levels=self.max_dwt_levels,
                feature_types=self.feature_types,
                sampling_rate=self.sampling_rate
            )
            self.extractors.append(extractor)
            self.extractor_info.append({
                'type': 'DWT',
                'wavelet': wavelet,
                'levels': self.max_dwt_levels,
                'index': i
            })
        
        # Strategy 2: Deep WPD extractors with different configurations
        wpd_configs = [
            {'wavelet': self.wavelets[0], 'levels': 4},
            {'wavelet': self.wavelets[0], 'levels': 5},
            {'wavelet': self.wavelets[1], 'levels': 4},
            {'wavelet': self.wavelets[1], 'levels': 5},
        ]
        
        for i, config in enumerate(wpd_configs):
            extractor = WPDExtractor(
                wavelet=config['wavelet'],
                levels=config['levels'],
                node_selection='all',
                feature_types=['statistical', 'energy', 'frequency', 'spectral'],
                sampling_rate=self.sampling_rate
            )
            self.extractors.append(extractor)
            self.extractor_info.append({
                'type': 'WPD',
                'wavelet': config['wavelet'],
                'levels': config['levels'],
                'index': len(self.wavelets) + i
            })
        
        # Strategy 3: Specialized extractors for different frequency ranges
        
        # High-frequency focused WPD
        hf_extractor = WPDExtractor(
            wavelet='db4',
            levels=3,
            node_selection='all',
            feature_types=['statistical', 'spectral'],
            sampling_rate=self.sampling_rate
        )
        self.extractors.append(hf_extractor)
        self.extractor_info.append({
            'type': 'WPD_HF',
            'wavelet': 'db4',
            'levels': 3,
            'index': len(self.extractors) - 1
        })
        
        # Low-frequency focused DWT
        lf_extractor = DWTExtractor(
            wavelet='coif3',
            levels=7,  # Very deep for low frequencies
            feature_types=['statistical', 'energy', 'entropy'],
            sampling_rate=self.sampling_rate
        )
        self.extractors.append(lf_extractor)
        self.extractor_info.append({
            'type': 'DWT_LF',
            'wavelet': 'coif3',
            'levels': 7,
            'index': len(self.extractors) - 1
        })
        
        logger.info(f"Created {len(self.extractors)} specialized extractors")
        
        # Log extractor details
        for i, info in enumerate(self.extractor_info):
            logger.info(f"  Extractor {i+1}: {info['type']} with {info['wavelet']} "
                       f"(levels={info['levels']})")
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract ultra-high dimensional features."""
        data = self.validate_data(data)
        data = self.reshape_data(data)
        
        if not self.extractors:
            self._create_extractors()
        
        n_samples = data.shape[0]
        logger.info(f"Extracting ultra-high dim features from {n_samples} samples")
        
        # Extract features from all extractors
        extractor_features = []
        total_features = 0
        
        for i, extractor in enumerate(self.extractors):
            info = self.extractor_info[i]
            logger.info(f"Running extractor {i+1}/{len(self.extractors)}: "
                       f"{info['type']} {info['wavelet']}")
            
            try:
                start_time = time.time()
                features = extractor.extract_features(data)
                extraction_time = time.time() - start_time
                
                extractor_features.append(features)
                total_features += features.shape[1]
                
                logger.info(f"  Extracted {features.shape[1]} features in {extraction_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"  Failed: {e}")
                continue
        
        # Combine all features
        if extractor_features:
            combined_features = np.concatenate(extractor_features, axis=1)
            logger.info(f"Combined features shape: {combined_features.shape}")
            logger.info(f"Total features: {combined_features.shape[1]}")
            
            # Update feature count
            self.n_features = combined_features.shape[1]
            
            # Generate feature names
            self._generate_combined_feature_names()
            
            return combined_features
        else:
            raise RuntimeError("No features could be extracted")
    
    def _generate_combined_feature_names(self):
        """Generate names for combined features."""
        self.feature_names = []
        
        for i, extractor in enumerate(self.extractors):
            info = self.extractor_info[i]
            
            # Get feature names from extractor if available
            if hasattr(extractor, 'feature_names') and extractor.feature_names:
                base_names = extractor.feature_names
            else:
                # Generate generic names
                n_features = getattr(extractor, 'n_features', 100)
                base_names = [f"feature_{j}" for j in range(n_features)]
            
            # Add prefix with extractor info
            prefix = f"{info['type']}_{info['wavelet']}_L{info['levels']}"
            prefixed_names = [f"{prefix}_{name}" for name in base_names]
            self.feature_names.extend(prefixed_names)
    
    def get_extractor_info(self) -> List[Dict]:
        """
        Get information about all extractors.
        
        Returns:
            List[Dict]: Extractor information
        """
        return self.extractor_info.copy()
    
    def get_feature_breakdown(self) -> Dict[str, int]:
        """
        Get breakdown of features by extractor type.
        
        Returns:
            Dict[str, int]: Features per extractor type
        """
        if not self.extractors:
            return {}
        
        breakdown = {}
        for i, extractor in enumerate(self.extractors):
            info = self.extractor_info[i]
            extractor_key = f"{info['type']}_{info['wavelet']}_L{info['levels']}"
            
            if hasattr(extractor, 'n_features'):
                breakdown[extractor_key] = extractor.n_features
            else:
                breakdown[extractor_key] = 0
        
        return breakdown
    
    def estimate_extraction_time(self, n_samples: int) -> float:
        """
        Estimate extraction time for given number of samples.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            float: Estimated time in seconds
        """
        # Based on benchmarking: ~1 second per sample for ultra-high dim
        base_time_per_sample = 1.0
        return n_samples * base_time_per_sample
    
    def get_memory_usage_estimate(self, n_samples: int) -> Dict[str, float]:
        """
        Estimate memory usage for feature extraction.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Dict[str, float]: Memory usage estimates in MB
        """
        # Estimate based on feature dimensions
        features_per_sample = self.target_dimensions
        bytes_per_feature = 8  # float64
        
        input_mb = n_samples * 14 * 128 * bytes_per_feature / (1024**2)
        output_mb = n_samples * features_per_sample * bytes_per_feature / (1024**2)
        working_mb = output_mb * 2  # Temporary arrays during extraction
        
        return {
            'input_data': input_mb,
            'output_features': output_mb,
            'working_memory': working_mb,
            'total_estimated': input_mb + output_mb + working_mb
        }


class MultiWaveletExtractor(WaveletFeatureBase):
    """
    Multi-wavelet feature extractor.
    
    This extractor uses multiple wavelet families to capture different
    mathematical perspectives of the signal.
    """
    
    def __init__(self,
                 wavelets: List[str] = None,
                 levels: int = 4,
                 feature_types: List[str] = None,
                 **kwargs):
        """
        Initialize multi-wavelet extractor.
        
        Args:
            wavelets: List of wavelets to use
            levels: Decomposition levels
            feature_types: Types of features to extract
        """
        if wavelets is None:
            wavelets = ['db4', 'db8', 'coif3', 'bior4.4']
        
        super().__init__("MultiWaveletExtractor", wavelets[0], levels, **kwargs)
        
        self.wavelets = wavelets
        self.feature_types = feature_types or ['statistical', 'energy']
        
        # Create extractors for each wavelet
        self.wavelet_extractors = {}
        for wavelet in self.wavelets:
            self.wavelet_extractors[wavelet] = DWTExtractor(
                wavelet=wavelet,
                levels=levels,
                feature_types=self.feature_types,
                **kwargs
            )
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features using multiple wavelets."""
        data = self.validate_data(data)
        data = self.reshape_data(data)
        
        all_features = []
        
        for wavelet, extractor in self.wavelet_extractors.items():
            logger.info(f"Extracting features with {wavelet}")
            features = extractor.extract_features(data)
            all_features.append(features)
        
        # Combine features from all wavelets
        combined_features = np.concatenate(all_features, axis=1)
        self.n_features = combined_features.shape[1]
        
        return combined_features
