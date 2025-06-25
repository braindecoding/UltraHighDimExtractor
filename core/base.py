"""
Base classes for UltraHighDimWaveletExtractor
============================================

This module provides base classes and interfaces for wavelet feature extraction.

Classes:
- FeatureExtractorInterface: Abstract interface for feature extractors
- WaveletFeatureBase: Base class for wavelet-based feature extractors
- WaveletAnalyzer: Utility class for wavelet analysis
"""

import numpy as np
import pywt
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FeatureExtractorInterface(ABC):
    """
    Abstract interface for feature extractors.
    
    This interface defines the standard methods that all feature extractors
    should implement for consistency and interoperability.
    """
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'FeatureExtractorInterface':
        """
        Fit the extractor to training data.
        
        Args:
            data: Training data
            
        Returns:
            self: Fitted extractor
        """
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data to features.
        
        Args:
            data: Input data
            
        Returns:
            np.ndarray: Extracted features
        """
        pass
    
    @abstractmethod
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform data.
        
        Args:
            data: Input data
            
        Returns:
            np.ndarray: Extracted features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features.
        
        Returns:
            List[str]: Feature names
        """
        pass


class WaveletFeatureBase(FeatureExtractorInterface):
    """
    Base class for wavelet-based feature extractors.
    
    This class provides common functionality for wavelet feature extraction
    including data validation, reshaping, and basic wavelet operations.
    """
    
    def __init__(self, 
                 name: str,
                 wavelet: str = 'db4',
                 levels: int = 4,
                 sampling_rate: float = 128.0,
                 **kwargs):
        """
        Initialize wavelet feature extractor.
        
        Args:
            name: Name of the extractor
            wavelet: Wavelet type
            levels: Number of decomposition levels
            sampling_rate: Sampling rate in Hz
        """
        self.name = name
        self.wavelet = wavelet
        self.levels = levels
        self.sampling_rate = sampling_rate
        
        # Feature extraction state
        self.is_fitted = False
        self.n_features = 0
        self.feature_names = []
        
        # Validate wavelet
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Wavelet '{wavelet}' not supported. Available: {pywt.wavelist()}")
        
        logger.info(f"Initialized {name} with wavelet={wavelet}, levels={levels}")
    
    def validate_data(self, data: np.ndarray) -> np.ndarray:
        """
        Validate input data format and content.
        
        Args:
            data: Input data
            
        Returns:
            np.ndarray: Validated data
            
        Raises:
            ValueError: If data format is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be numpy array")
        
        if data.ndim < 2 or data.ndim > 3:
            raise ValueError("Data must be 2D or 3D array")
        
        # Check for invalid values
        if np.isnan(data).any():
            logger.warning("Data contains NaN values")
        
        if np.isinf(data).any():
            logger.warning("Data contains infinite values")
        
        return data
    
    def reshape_data(self, data: np.ndarray) -> np.ndarray:
        """
        Reshape data to standard format (n_samples, n_channels, n_timepoints).
        
        Args:
            data: Input data
            
        Returns:
            np.ndarray: Reshaped data
        """
        if data.ndim == 2:
            # Assume (n_samples, n_features) -> reshape to (n_samples, n_channels, n_timepoints)
            n_samples, n_features = data.shape
            
            # Try to infer channel and timepoint dimensions
            if n_features == 1792:  # 14 channels * 128 timepoints
                data = data.reshape(n_samples, 14, 128)
            elif n_features % 14 == 0:
                n_timepoints = n_features // 14
                data = data.reshape(n_samples, 14, n_timepoints)
            else:
                raise ValueError(f"Cannot reshape data with {n_features} features")
        
        elif data.ndim == 3:
            # Already in correct format
            pass
        
        else:
            raise ValueError(f"Cannot handle {data.ndim}D data")
        
        return data
    
    def fit(self, data: np.ndarray) -> 'WaveletFeatureBase':
        """
        Fit the extractor to training data.
        
        Args:
            data: Training data
            
        Returns:
            self: Fitted extractor
        """
        logger.info(f"Fitting {self.name}")
        
        # Validate and reshape data
        data = self.validate_data(data)
        data = self.reshape_data(data)
        
        # Perform fitting (can be overridden by subclasses)
        self._fit_implementation(data)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data to features.
        
        Args:
            data: Input data
            
        Returns:
            np.ndarray: Extracted features
        """
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before transform")
        
        return self.extract_features(data)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform data.
        
        Args:
            data: Input data
            
        Returns:
            np.ndarray: Extracted features
        """
        return self.fit(data).transform(data)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features.
        
        Returns:
            List[str]: Feature names
        """
        return self.feature_names.copy()
    
    def _fit_implementation(self, data: np.ndarray) -> None:
        """
        Implementation-specific fitting logic.
        
        Args:
            data: Training data
            
        Note:
            This method can be overridden by subclasses for custom fitting logic.
        """
        pass
    
    @abstractmethod
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features from data.
        
        Args:
            data: Input data
            
        Returns:
            np.ndarray: Extracted features
        """
        pass


class WaveletAnalyzer:
    """
    Utility class for wavelet analysis and selection.
    
    This class provides methods for analyzing wavelets and selecting
    optimal parameters for feature extraction.
    """
    
    @staticmethod
    def get_available_wavelets() -> Dict[str, List[str]]:
        """
        Get available wavelets organized by family.

        Returns:
            Dict[str, List[str]]: Wavelets by family
        """
        families = {}

        for wavelet in pywt.wavelist():
            try:
                # Only process discrete wavelets
                w = pywt.Wavelet(wavelet)
                family = w.family_name
                if family not in families:
                    families[family] = []
                families[family].append(wavelet)
            except ValueError:
                # Skip continuous wavelets
                continue

        return families
    
    @staticmethod
    def analyze_wavelet_properties(wavelet: str) -> Dict:
        """
        Analyze properties of a specific wavelet.
        
        Args:
            wavelet: Wavelet name
            
        Returns:
            Dict: Wavelet properties
        """
        w = pywt.Wavelet(wavelet)
        
        return {
            'name': wavelet,
            'family': w.family_name,
            'short_name': w.short_name,
            'orthogonal': w.orthogonal,
            'biorthogonal': w.biorthogonal,
            'symmetry': w.symmetry,
            'vanishing_moments_psi': w.vanishing_moments_psi,
            'vanishing_moments_phi': w.vanishing_moments_phi
        }
    
    @staticmethod
    def recommend_wavelets_for_eeg() -> List[str]:
        """
        Recommend wavelets suitable for EEG analysis.
        
        Returns:
            List[str]: Recommended wavelets
        """
        return ['db4', 'db6', 'db8', 'coif3', 'coif5', 'bior4.4', 'sym4', 'sym8']
    
    @staticmethod
    def estimate_optimal_levels(signal_length: int, wavelet: str) -> int:
        """
        Estimate optimal decomposition levels for a signal.
        
        Args:
            signal_length: Length of the signal
            wavelet: Wavelet name
            
        Returns:
            int: Recommended number of levels
        """
        max_levels = pywt.dwt_max_level(signal_length, wavelet)
        
        # Use conservative estimate (usually max_levels - 1 or 2)
        optimal_levels = max(1, min(max_levels - 1, 6))
        
        return optimal_levels
    
    @staticmethod
    def compute_frequency_bands(levels: int, sampling_rate: float) -> List[Tuple[float, float]]:
        """
        Compute frequency bands for wavelet decomposition levels.
        
        Args:
            levels: Number of decomposition levels
            sampling_rate: Sampling rate in Hz
            
        Returns:
            List[Tuple[float, float]]: Frequency bands (low, high)
        """
        nyquist = sampling_rate / 2
        bands = []
        
        for level in range(1, levels + 1):
            high_freq = nyquist / (2 ** (level - 1))
            low_freq = nyquist / (2 ** level)
            bands.append((low_freq, high_freq))
        
        # Add approximation band
        bands.append((0, nyquist / (2 ** levels)))
        
        return bands
