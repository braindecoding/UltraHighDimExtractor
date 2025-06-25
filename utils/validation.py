"""
Data Validation Utilities
=========================

This module provides comprehensive data validation functions for EEG data
and extracted features.

Functions:
- validate_eeg_data: Validate raw EEG data format and content
- validate_features: Validate extracted features
- DataValidator: Comprehensive data validation class
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def validate_eeg_data(data: np.ndarray, 
                     expected_channels: int = 14,
                     expected_timepoints: int = 128,
                     sampling_rate: float = 128.0) -> np.ndarray:
    """
    Validate raw EEG data format and content.
    
    Args:
        data: EEG data array
        expected_channels: Expected number of channels
        expected_timepoints: Expected number of timepoints
        sampling_rate: Expected sampling rate
        
    Returns:
        np.ndarray: Validated data
        
    Raises:
        ValueError: If data format is invalid
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be numpy array")
    
    # Check dimensions
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")
    
    # Reshape if needed
    if data.ndim == 2:
        n_samples, n_features = data.shape
        if n_features == expected_channels * expected_timepoints:
            data = data.reshape(n_samples, expected_channels, expected_timepoints)
        else:
            raise ValueError(f"Cannot reshape 2D data with {n_features} features")
    
    # Validate 3D shape
    if data.ndim == 3:
        n_samples, n_channels, n_timepoints = data.shape
        
        if n_channels != expected_channels:
            logger.warning(f"Expected {expected_channels} channels, got {n_channels}")
        
        if n_timepoints != expected_timepoints:
            logger.warning(f"Expected {expected_timepoints} timepoints, got {n_timepoints}")
    
    # Check for invalid values
    if np.isnan(data).any():
        n_nan = np.sum(np.isnan(data))
        logger.warning(f"Data contains {n_nan} NaN values")
    
    if np.isinf(data).any():
        n_inf = np.sum(np.isinf(data))
        logger.warning(f"Data contains {n_inf} infinite values")
    
    # Check amplitude ranges (typical EEG is -100 to +100 µV)
    data_min, data_max = np.min(data), np.max(data)
    if data_max > 1000 or data_min < -1000:
        logger.warning(f"Unusual amplitude range: [{data_min:.2f}, {data_max:.2f}]")
    
    return data


def validate_features(features: np.ndarray,
                     expected_samples: Optional[int] = None,
                     min_features: int = 1000) -> np.ndarray:
    """
    Validate extracted features.
    
    Args:
        features: Feature array
        expected_samples: Expected number of samples
        min_features: Minimum expected number of features
        
    Returns:
        np.ndarray: Validated features
        
    Raises:
        ValueError: If features are invalid
    """
    if not isinstance(features, np.ndarray):
        raise ValueError("Features must be numpy array")
    
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D, got {features.ndim}D")
    
    n_samples, n_features = features.shape
    
    # Check sample count
    if expected_samples is not None and n_samples != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, got {n_samples}")
    
    # Check feature count
    if n_features < min_features:
        logger.warning(f"Low feature count: {n_features} < {min_features}")
    
    # Check for invalid values
    n_nan = np.sum(np.isnan(features))
    if n_nan > 0:
        logger.warning(f"Features contain {n_nan} NaN values")
    
    n_inf = np.sum(np.isinf(features))
    if n_inf > 0:
        logger.warning(f"Features contain {n_inf} infinite values")
    
    # Check for zero variance features
    feature_vars = np.var(features, axis=0)
    n_zero_var = np.sum(feature_vars == 0)
    if n_zero_var > 0:
        logger.warning(f"Features contain {n_zero_var} zero-variance features")
    
    return features


class DataValidator:
    """
    Comprehensive data validation class.
    
    This class provides detailed validation and quality assessment
    for EEG data and extracted features.
    """
    
    def __init__(self,
                 expected_channels: int = 14,
                 expected_timepoints: int = 128,
                 sampling_rate: float = 128.0):
        """
        Initialize data validator.
        
        Args:
            expected_channels: Expected number of EEG channels
            expected_timepoints: Expected number of timepoints
            sampling_rate: Expected sampling rate
        """
        self.expected_channels = expected_channels
        self.expected_timepoints = expected_timepoints
        self.sampling_rate = sampling_rate
    
    def validate_eeg_comprehensive(self, data: np.ndarray) -> Dict:
        """
        Comprehensive EEG data validation.
        
        Args:
            data: EEG data array
            
        Returns:
            Dict: Validation report
        """
        report = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {},
            'recommendations': []
        }
        
        try:
            # Basic validation
            data = validate_eeg_data(data, self.expected_channels, 
                                   self.expected_timepoints, self.sampling_rate)
            
            # Detailed statistics
            report['statistics'] = self._compute_eeg_statistics(data)
            
            # Quality checks
            self._check_eeg_quality(data, report)
            
        except Exception as e:
            report['valid'] = False
            report['errors'].append(str(e))
        
        return report
    
    def validate_features_comprehensive(self, features: np.ndarray) -> Dict:
        """
        Comprehensive feature validation.
        
        Args:
            features: Feature array
            
        Returns:
            Dict: Validation report
        """
        report = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {},
            'recommendations': []
        }
        
        try:
            # Basic validation
            features = validate_features(features)
            
            # Detailed statistics
            report['statistics'] = self._compute_feature_statistics(features)
            
            # Quality checks
            self._check_feature_quality(features, report)
            
        except Exception as e:
            report['valid'] = False
            report['errors'].append(str(e))
        
        return report
    
    def _compute_eeg_statistics(self, data: np.ndarray) -> Dict:
        """Compute detailed EEG statistics."""
        n_samples, n_channels, n_timepoints = data.shape
        
        stats = {
            'shape': data.shape,
            'n_samples': n_samples,
            'n_channels': n_channels,
            'n_timepoints': n_timepoints,
            'duration_seconds': n_timepoints / self.sampling_rate,
            'amplitude_range': [float(np.min(data)), float(np.max(data))],
            'mean_amplitude': float(np.mean(data)),
            'std_amplitude': float(np.std(data)),
            'channel_means': np.mean(data, axis=(0, 2)).tolist(),
            'channel_stds': np.std(data, axis=(0, 2)).tolist(),
            'n_nan': int(np.sum(np.isnan(data))),
            'n_inf': int(np.sum(np.isinf(data)))
        }
        
        return stats
    
    def _compute_feature_statistics(self, features: np.ndarray) -> Dict:
        """Compute detailed feature statistics."""
        n_samples, n_features = features.shape
        
        feature_vars = np.var(features, axis=0)
        feature_means = np.mean(features, axis=0)
        
        stats = {
            'shape': features.shape,
            'n_samples': n_samples,
            'n_features': n_features,
            'value_range': [float(np.min(features)), float(np.max(features))],
            'mean_value': float(np.mean(features)),
            'std_value': float(np.std(features)),
            'n_nan': int(np.sum(np.isnan(features))),
            'n_inf': int(np.sum(np.isinf(features))),
            'n_zero_variance': int(np.sum(feature_vars == 0)),
            'min_variance': float(np.min(feature_vars)),
            'max_variance': float(np.max(feature_vars)),
            'mean_variance': float(np.mean(feature_vars))
        }
        
        return stats
    
    def _check_eeg_quality(self, data: np.ndarray, report: Dict):
        """Check EEG data quality and add warnings/recommendations."""
        n_samples, n_channels, n_timepoints = data.shape
        
        # Check amplitude range
        data_min, data_max = np.min(data), np.max(data)
        if data_max > 500 or data_min < -500:
            report['warnings'].append(f"Unusual amplitude range: [{data_min:.2f}, {data_max:.2f}]")
            report['recommendations'].append("Consider checking electrode impedances and preprocessing")
        
        # Check for flat channels
        channel_vars = np.var(data, axis=(0, 2))
        flat_channels = np.where(channel_vars < 0.01)[0]
        if len(flat_channels) > 0:
            report['warnings'].append(f"Flat channels detected: {flat_channels.tolist()}")
            report['recommendations'].append("Check electrode connections")
        
        # Check for excessive noise
        channel_stds = np.std(data, axis=(0, 2))
        noisy_channels = np.where(channel_stds > 100)[0]
        if len(noisy_channels) > 0:
            report['warnings'].append(f"Noisy channels detected: {noisy_channels.tolist()}")
            report['recommendations'].append("Consider artifact removal or channel interpolation")
        
        # Check for missing data
        if np.isnan(data).any():
            report['warnings'].append("Missing data (NaN values) detected")
            report['recommendations'].append("Apply interpolation or remove affected samples")
    
    def _check_feature_quality(self, features: np.ndarray, report: Dict):
        """Check feature quality and add warnings/recommendations."""
        n_samples, n_features = features.shape
        
        # Check feature dimensionality
        if n_features < 10000:
            report['warnings'].append(f"Low feature dimensionality: {n_features}")
            report['recommendations'].append("Consider using more comprehensive feature extraction")
        
        # Check for zero variance features
        feature_vars = np.var(features, axis=0)
        n_zero_var = np.sum(feature_vars == 0)
        if n_zero_var > n_features * 0.1:  # More than 10%
            report['warnings'].append(f"High proportion of zero-variance features: {n_zero_var}/{n_features}")
            report['recommendations'].append("Apply feature selection to remove uninformative features")
        
        # Check for extreme values
        if np.isinf(features).any():
            report['warnings'].append("Infinite values detected in features")
            report['recommendations'].append("Check feature extraction parameters")
        
        # Check feature distribution
        feature_means = np.mean(features, axis=0)
        if np.std(feature_means) > 1000:
            report['warnings'].append("High variance in feature means")
            report['recommendations'].append("Consider feature normalization")
    
    def generate_validation_summary(self, eeg_report: Dict, feature_report: Dict) -> str:
        """
        Generate human-readable validation summary.
        
        Args:
            eeg_report: EEG validation report
            feature_report: Feature validation report
            
        Returns:
            str: Validation summary
        """
        summary = []
        summary.append("🔍 DATA VALIDATION SUMMARY")
        summary.append("=" * 50)
        
        # EEG validation
        summary.append("\n📊 EEG Data Validation:")
        if eeg_report['valid']:
            summary.append("✅ EEG data format is valid")
            stats = eeg_report['statistics']
            summary.append(f"   Shape: {stats['shape']}")
            summary.append(f"   Duration: {stats['duration_seconds']:.2f}s")
            summary.append(f"   Amplitude range: [{stats['amplitude_range'][0]:.2f}, {stats['amplitude_range'][1]:.2f}]")
        else:
            summary.append("❌ EEG data validation failed")
            for error in eeg_report['errors']:
                summary.append(f"   Error: {error}")
        
        if eeg_report['warnings']:
            summary.append("⚠️  Warnings:")
            for warning in eeg_report['warnings']:
                summary.append(f"   - {warning}")
        
        # Feature validation
        summary.append("\n🎯 Feature Validation:")
        if feature_report['valid']:
            summary.append("✅ Features are valid")
            stats = feature_report['statistics']
            summary.append(f"   Shape: {stats['shape']}")
            summary.append(f"   Feature count: {stats['n_features']:,}")
            summary.append(f"   Zero variance: {stats['n_zero_variance']}")
        else:
            summary.append("❌ Feature validation failed")
            for error in feature_report['errors']:
                summary.append(f"   Error: {error}")
        
        if feature_report['warnings']:
            summary.append("⚠️  Warnings:")
            for warning in feature_report['warnings']:
                summary.append(f"   - {warning}")
        
        # Recommendations
        all_recommendations = eeg_report.get('recommendations', []) + feature_report.get('recommendations', [])
        if all_recommendations:
            summary.append("\n💡 Recommendations:")
            for rec in set(all_recommendations):  # Remove duplicates
                summary.append(f"   - {rec}")
        
        return "\n".join(summary)
