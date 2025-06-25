#!/usr/bin/env python3
"""
Wavelet Feature Extraction Base Classes
=======================================

This module provides base classes and utilities specifically for wavelet-based
feature extraction from EEG signals. It focuses on comprehensive wavelet analysis
with different decomposition methods.

Author: EEG Research Team
Date: 2025-06-25
"""

import numpy as np
import pywt
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union
import logging
from scipy.stats import entropy
from scipy import signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WaveletFeatureBase(ABC):
    """
    Abstract base class for wavelet feature extractors.
    
    This class provides common functionality for all wavelet-based feature
    extraction methods including data validation, wavelet selection, and
    feature naming conventions.
    """
    
    def __init__(self, 
                 name: str,
                 wavelet: str = 'db4',
                 levels: int = 4,
                 sampling_rate: float = 128.0,
                 **kwargs):
        """
        Initialize the wavelet feature extractor.
        
        Args:
            name (str): Name of the extractor
            wavelet (str): Wavelet type (e.g., 'db4', 'haar', 'coif2')
            levels (int): Number of decomposition levels
            sampling_rate (float): Sampling rate in Hz
            **kwargs: Additional parameters
        """
        self.name = name
        self.wavelet = wavelet
        self.levels = levels
        self.sampling_rate = sampling_rate
        self.params = kwargs
        
        # Validate wavelet
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Wavelet '{wavelet}' not supported. Available: {pywt.wavelist()}")
        
        # Initialize feature tracking
        self.feature_names = []
        self.n_features = 0
        self.is_fitted = False
        
        logger.info(f"Initialized {name} with wavelet={wavelet}, levels={levels}")
    
    @abstractmethod
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract wavelet features from EEG data.
        
        Args:
            data (np.ndarray): EEG data
            
        Returns:
            np.ndarray: Extracted features
        """
        pass
    
    def validate_data(self, data: np.ndarray) -> np.ndarray:
        """
        Validate and preprocess input data.
        
        Args:
            data (np.ndarray): Input EEG data
            
        Returns:
            np.ndarray: Validated data
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        
        if data.ndim < 2:
            raise ValueError("Data must have at least 2 dimensions")
        
        # Handle NaN and infinite values
        if np.isnan(data).any():
            logger.warning("NaN values detected, replacing with zeros")
            data = np.nan_to_num(data, nan=0.0)
        
        if np.isinf(data).any():
            logger.warning("Infinite values detected, clipping to finite range")
            data = np.nan_to_num(data, posinf=1e6, neginf=-1e6)
        
        return data
    
    def reshape_data(self, data: np.ndarray) -> np.ndarray:
        """
        Reshape data to standard format (n_samples, n_channels, n_timepoints).
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Reshaped data
        """
        if data.ndim == 2:
            # Assume flattened data: (n_samples, n_channels * n_timepoints)
            n_samples = data.shape[0]
            n_features = data.shape[1]
            
            # Try to reshape assuming 14 channels
            n_channels = 14
            n_timepoints = n_features // n_channels
            
            if n_features % n_channels == 0:
                data = data.reshape(n_samples, n_channels, n_timepoints)
                logger.info(f"Reshaped data to ({n_samples}, {n_channels}, {n_timepoints})")
            else:
                # If not divisible, treat as single channel
                data = data.reshape(n_samples, 1, n_features)
                logger.info(f"Reshaped data to single channel: ({n_samples}, 1, {n_features})")
        
        return data
    
    def get_wavelet_info(self) -> Dict:
        """Get information about the selected wavelet."""
        try:
            wavelet_obj = pywt.Wavelet(self.wavelet)
            return {
                'name': self.wavelet,
                'family': wavelet_obj.family_name,
                'biorthogonal': wavelet_obj.biorthogonal,
                'orthogonal': wavelet_obj.orthogonal,
                'symmetry': wavelet_obj.symmetry,
                'vanishing_moments_psi': getattr(wavelet_obj, 'vanishing_moments_psi', None),
                'vanishing_moments_phi': getattr(wavelet_obj, 'vanishing_moments_phi', None)
            }
        except Exception as e:
            logger.warning(f"Could not get wavelet info: {e}")
            return {'name': self.wavelet}
    
    def compute_statistical_features(self, coefficients: np.ndarray) -> List[float]:
        """
        Compute comprehensive statistical features from wavelet coefficients.
        
        Args:
            coefficients (np.ndarray): Wavelet coefficients
            
        Returns:
            List[float]: Statistical features
        """
        if len(coefficients) == 0:
            return [0.0] * 12  # Return zeros for empty coefficients
        
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(coefficients),              # Mean
            np.std(coefficients),               # Standard deviation
            np.var(coefficients),               # Variance
            np.min(coefficients),               # Minimum
            np.max(coefficients),               # Maximum
            np.median(coefficients),            # Median
            np.ptp(coefficients),               # Peak-to-peak
            np.mean(np.abs(coefficients)),      # Mean absolute value
            np.sqrt(np.mean(coefficients**2))   # RMS
        ])
        
        # Energy
        energy = np.sum(coefficients**2)
        features.append(energy)
        
        # Entropy
        coeff_abs = np.abs(coefficients)
        if np.sum(coeff_abs) > 0:
            prob = coeff_abs / np.sum(coeff_abs)
            prob = prob[prob > 0]  # Remove zeros
            features.append(entropy(prob))
        else:
            features.append(0.0)
        
        # Relative energy (normalized by total energy)
        total_energy = np.sum(np.abs(coefficients))
        if total_energy > 0:
            features.append(energy / total_energy)
        else:
            features.append(0.0)
        
        return features
    
    def get_statistical_feature_names(self, prefix: str) -> List[str]:
        """Get names for statistical features."""
        return [
            f"{prefix}_mean", f"{prefix}_std", f"{prefix}_var",
            f"{prefix}_min", f"{prefix}_max", f"{prefix}_median",
            f"{prefix}_ptp", f"{prefix}_mean_abs", f"{prefix}_rms",
            f"{prefix}_energy", f"{prefix}_entropy", f"{prefix}_rel_energy"
        ]
    
    def fit(self, data: np.ndarray) -> 'WaveletFeatureBase':
        """
        Fit the extractor (if needed).
        
        Args:
            data (np.ndarray): Training data
            
        Returns:
            self: Fitted extractor
        """
        logger.info(f"Fitting {self.name}")
        self.is_fitted = True
        return self
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform data.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Extracted features
        """
        return self.fit(data).extract_features(data)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names.copy()
    
    def get_params(self) -> Dict:
        """Get parameters."""
        return {
            'wavelet': self.wavelet,
            'levels': self.levels,
            'sampling_rate': self.sampling_rate,
            **self.params
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(wavelet='{self.wavelet}', levels={self.levels})"


class WaveletAnalyzer:
    """
    Utility class for wavelet analysis and visualization.
    
    This class provides methods for analyzing wavelet properties,
    optimal parameter selection, and visualization of decompositions.
    """
    
    @staticmethod
    def list_available_wavelets() -> Dict[str, List[str]]:
        """List all available wavelets by family."""
        wavelets_by_family = {}
        
        for wavelet in pywt.wavelist():
            try:
                w = pywt.Wavelet(wavelet)
                family = w.family_name
                if family not in wavelets_by_family:
                    wavelets_by_family[family] = []
                wavelets_by_family[family].append(wavelet)
            except:
                # Handle wavelets that can't be instantiated
                if 'other' not in wavelets_by_family:
                    wavelets_by_family['other'] = []
                wavelets_by_family['other'].append(wavelet)
        
        return wavelets_by_family
    
    @staticmethod
    def recommend_wavelet(signal_type: str = 'eeg') -> List[str]:
        """
        Recommend wavelets based on signal type.
        
        Args:
            signal_type (str): Type of signal ('eeg', 'ecg', 'general')
            
        Returns:
            List[str]: Recommended wavelets
        """
        recommendations = {
            'eeg': ['db4', 'db6', 'coif3', 'bior4.4', 'sym4'],
            'ecg': ['db6', 'db8', 'coif5', 'bior6.8'],
            'general': ['db4', 'haar', 'coif2', 'bior2.2']
        }
        
        return recommendations.get(signal_type, recommendations['general'])
    
    @staticmethod
    def estimate_optimal_levels(signal_length: int, 
                              wavelet: str = 'db4',
                              min_coeff_length: int = 4) -> int:
        """
        Estimate optimal number of decomposition levels.
        
        Args:
            signal_length (int): Length of the signal
            wavelet (str): Wavelet type
            min_coeff_length (int): Minimum coefficient length
            
        Returns:
            int: Recommended number of levels
        """
        try:
            max_levels = pywt.dwt_max_level(signal_length, wavelet)
            
            # Ensure we don't go too deep
            optimal_levels = min(max_levels, 6)  # Cap at 6 levels
            
            # Check if coefficients would be too short
            test_levels = optimal_levels
            while test_levels > 1:
                test_signal = np.random.randn(signal_length)
                coeffs = pywt.wavedec(test_signal, wavelet, level=test_levels)
                if all(len(c) >= min_coeff_length for c in coeffs):
                    break
                test_levels -= 1
            
            return max(1, test_levels)
            
        except Exception as e:
            logger.warning(f"Could not estimate optimal levels: {e}")
            return 4  # Default fallback
    
    @staticmethod
    def analyze_frequency_bands(coefficients: List[np.ndarray], 
                              sampling_rate: float = 128.0) -> Dict[str, Dict]:
        """
        Analyze frequency content of wavelet coefficients.
        
        Args:
            coefficients (List[np.ndarray]): Wavelet coefficients from wavedec
            sampling_rate (float): Sampling rate in Hz
            
        Returns:
            Dict: Frequency band analysis
        """
        analysis = {}
        
        # Calculate frequency ranges for each level
        nyquist = sampling_rate / 2
        
        for i, coeff in enumerate(coefficients):
            if i == 0:
                # Approximation coefficients (lowest frequencies)
                freq_range = (0, nyquist / (2 ** len(coefficients)))
                band_name = f"Approximation_L{len(coefficients)-1}"
            else:
                # Detail coefficients
                level = len(coefficients) - i
                freq_low = nyquist / (2 ** (level + 1))
                freq_high = nyquist / (2 ** level)
                freq_range = (freq_low, freq_high)
                band_name = f"Detail_L{level}"
            
            analysis[band_name] = {
                'frequency_range': freq_range,
                'coefficient_length': len(coeff),
                'energy': np.sum(coeff**2),
                'mean_amplitude': np.mean(np.abs(coeff)),
                'std_amplitude': np.std(np.abs(coeff))
            }
        
        return analysis
