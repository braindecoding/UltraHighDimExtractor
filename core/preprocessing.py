#!/usr/bin/env python3
"""
Comprehensive EEG Preprocessing Pipeline
=======================================

This module provides comprehensive preprocessing for raw EEG data before
wavelet feature extraction, specifically optimized for image reconstruction tasks.

Author: EEG Research Team
Date: 2025-06-25
"""

import numpy as np
from scipy import signal, stats
from scipy.signal import butter, filtfilt, detrend
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """
    Comprehensive EEG preprocessing pipeline for wavelet feature extraction.
    
    This class implements state-of-the-art EEG preprocessing techniques
    optimized for preserving information relevant to image reconstruction.
    """
    
    def __init__(self,
                 sampling_rate: float = 128.0,
                 lowpass_freq: float = 50.0,
                 highpass_freq: float = 0.5,
                 notch_freq: float = 50.0,
                 artifact_threshold: float = 5.0,
                 baseline_method: str = 'detrend',
                 scaling_method: str = 'robust',
                 **kwargs):
        """
        Initialize EEG preprocessor.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
            lowpass_freq: Low-pass filter cutoff frequency
            highpass_freq: High-pass filter cutoff frequency  
            notch_freq: Notch filter frequency (50/60 Hz)
            artifact_threshold: Threshold for artifact detection (in std)
            baseline_method: Baseline correction method
            scaling_method: Scaling method ('standard', 'robust', 'minmax')
        """
        self.sampling_rate = sampling_rate
        self.lowpass_freq = lowpass_freq
        self.highpass_freq = highpass_freq
        self.notch_freq = notch_freq
        self.artifact_threshold = artifact_threshold
        self.baseline_method = baseline_method
        self.scaling_method = scaling_method
        
        # Initialize scalers
        self.channel_scalers = {}
        self.is_fitted = False
        
        logger.info(f"EEG Preprocessor initialized: {lowpass_freq}Hz LP, {highpass_freq}Hz HP")
    
    def fit(self, data: np.ndarray) -> 'EEGPreprocessor':
        """
        Fit the preprocessor to training data.
        
        Args:
            data: Raw EEG data (n_samples, n_channels, n_timepoints)
            
        Returns:
            self: Fitted preprocessor
        """
        logger.info("Fitting EEG preprocessor...")
        
        # Validate input
        data = self._validate_input(data)
        
        # Fit scalers for each channel
        n_samples, n_channels, n_timepoints = data.shape
        
        for ch in range(n_channels):
            # Flatten channel data across all samples
            channel_data = data[:, ch, :].flatten()
            
            # Remove obvious artifacts before fitting scaler
            clean_data = self._remove_extreme_artifacts(channel_data)
            
            # Fit scaler
            if self.scaling_method == 'standard':
                scaler = StandardScaler()
            elif self.scaling_method == 'robust':
                scaler = RobustScaler()
            else:  # minmax
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            
            scaler.fit(clean_data.reshape(-1, 1))
            self.channel_scalers[ch] = scaler
        
        self.is_fitted = True
        logger.info("EEG preprocessor fitted successfully")
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to EEG data.
        
        Args:
            data: Raw EEG data (n_samples, n_channels, n_timepoints)
            
        Returns:
            np.ndarray: Preprocessed EEG data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.info("Applying EEG preprocessing...")
        
        # Validate input
        data = self._validate_input(data)
        processed_data = data.copy()
        
        n_samples, n_channels, n_timepoints = data.shape
        
        # Process each sample
        for sample_idx in range(n_samples):
            for ch_idx in range(n_channels):
                signal_data = processed_data[sample_idx, ch_idx, :]
                
                # Step 1: Baseline correction
                signal_data = self._baseline_correction(signal_data)
                
                # Step 2: Filtering
                signal_data = self._apply_filters(signal_data)
                
                # Step 3: Artifact removal
                signal_data = self._artifact_removal(signal_data)
                
                # Step 4: Scaling
                signal_data = self._apply_scaling(signal_data, ch_idx)
                
                processed_data[sample_idx, ch_idx, :] = signal_data
        
        logger.info("EEG preprocessing completed")
        return processed_data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(data).transform(data)
    
    def _validate_input(self, data: np.ndarray) -> np.ndarray:
        """Validate and prepare input data."""
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be numpy array")
        
        if data.ndim == 2:
            # Assume (n_samples, n_features) - reshape to (n_samples, n_channels, n_timepoints)
            n_samples, n_features = data.shape
            n_channels = 14  # Assume 14 channels
            n_timepoints = n_features // n_channels
            
            if n_features % n_channels == 0:
                data = data.reshape(n_samples, n_channels, n_timepoints)
            else:
                raise ValueError(f"Cannot reshape {data.shape} to (samples, 14, timepoints)")
        
        elif data.ndim != 3:
            raise ValueError("Data must be 2D or 3D array")
        
        # Check for NaN/inf values
        if np.isnan(data).any() or np.isinf(data).any():
            logger.warning("NaN/inf values detected, replacing with interpolation")
            data = self._handle_missing_values(data)
        
        return data
    
    def _baseline_correction(self, signal: np.ndarray) -> np.ndarray:
        """Apply baseline correction."""
        if self.baseline_method == 'detrend':
            # Remove linear trend
            return detrend(signal, type='linear')
        
        elif self.baseline_method == 'mean':
            # Remove mean
            return signal - np.mean(signal)
        
        elif self.baseline_method == 'median':
            # Remove median
            return signal - np.median(signal)
        
        elif self.baseline_method == 'highpass':
            # Very low frequency highpass (0.1 Hz)
            return self._highpass_filter(signal, 0.1)
        
        else:
            return signal
    
    def _apply_filters(self, signal: np.ndarray) -> np.ndarray:
        """Apply frequency domain filters."""
        filtered_signal = signal.copy()
        
        # High-pass filter (remove DC and very low frequencies)
        if self.highpass_freq > 0:
            filtered_signal = self._highpass_filter(filtered_signal, self.highpass_freq)
        
        # Low-pass filter (anti-aliasing and noise reduction)
        if self.lowpass_freq < self.sampling_rate / 2:
            filtered_signal = self._lowpass_filter(filtered_signal, self.lowpass_freq)
        
        # Notch filter (remove power line interference)
        if self.notch_freq > 0:
            filtered_signal = self._notch_filter(filtered_signal, self.notch_freq)
        
        return filtered_signal
    
    def _highpass_filter(self, signal: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply high-pass Butterworth filter."""
        try:
            nyquist = self.sampling_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            if normalized_cutoff >= 1.0:
                logger.warning(f"Highpass cutoff {cutoff} too high, skipping")
                return signal
            
            b, a = butter(4, normalized_cutoff, btype='high')
            return filtfilt(b, a, signal)
        except Exception as e:
            logger.warning(f"Highpass filter failed: {e}")
            return signal
    
    def _lowpass_filter(self, signal: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply low-pass Butterworth filter."""
        try:
            nyquist = self.sampling_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            if normalized_cutoff >= 1.0:
                logger.warning(f"Lowpass cutoff {cutoff} too high, skipping")
                return signal
            
            b, a = butter(4, normalized_cutoff, btype='low')
            return filtfilt(b, a, signal)
        except Exception as e:
            logger.warning(f"Lowpass filter failed: {e}")
            return signal
    
    def _notch_filter(self, signal_data: np.ndarray, freq: float, quality: float = 30.0) -> np.ndarray:
        """Apply notch filter for power line interference."""
        try:
            from scipy.signal import iirnotch

            nyquist = self.sampling_rate / 2
            normalized_freq = freq / nyquist

            if normalized_freq >= 1.0:
                logger.warning(f"Notch frequency {freq} too high, skipping")
                return signal_data

            b, a = iirnotch(normalized_freq, quality)
            return filtfilt(b, a, signal_data)
        except Exception as e:
            logger.warning(f"Notch filter failed: {e}")
            return signal_data
    
    def _artifact_removal(self, signal: np.ndarray) -> np.ndarray:
        """Remove artifacts using statistical methods."""
        # Method 1: Clip extreme values
        signal_std = np.std(signal)
        signal_mean = np.mean(signal)
        
        lower_bound = signal_mean - self.artifact_threshold * signal_std
        upper_bound = signal_mean + self.artifact_threshold * signal_std
        
        clipped_signal = np.clip(signal, lower_bound, upper_bound)
        
        # Method 2: Interpolate extreme artifacts
        artifact_mask = (signal < lower_bound) | (signal > upper_bound)
        
        if np.any(artifact_mask):
            # Simple linear interpolation for artifacts
            clean_indices = np.where(~artifact_mask)[0]
            artifact_indices = np.where(artifact_mask)[0]
            
            if len(clean_indices) > 1:
                interpolated_values = np.interp(artifact_indices, clean_indices, 
                                              signal[clean_indices])
                clipped_signal[artifact_mask] = interpolated_values
        
        return clipped_signal
    
    def _apply_scaling(self, signal: np.ndarray, channel_idx: int) -> np.ndarray:
        """Apply channel-specific scaling."""
        try:
            scaler = self.channel_scalers[channel_idx]
            scaled_signal = scaler.transform(signal.reshape(-1, 1)).flatten()
            return scaled_signal
        except Exception as e:
            logger.warning(f"Scaling failed for channel {channel_idx}: {e}")
            return signal
    
    def _remove_extreme_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Remove extreme artifacts for scaler fitting."""
        # Use robust statistics
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        
        # Remove outliers beyond 3*IQR
        lower_bound = q25 - 3 * iqr
        upper_bound = q75 + 3 * iqr
        
        mask = (data >= lower_bound) & (data <= upper_bound)
        return data[mask]
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle NaN and infinite values."""
        # Replace NaN with interpolation
        for sample_idx in range(data.shape[0]):
            for ch_idx in range(data.shape[1]):
                signal = data[sample_idx, ch_idx, :]
                
                if np.isnan(signal).any() or np.isinf(signal).any():
                    # Find valid indices
                    valid_mask = np.isfinite(signal)
                    
                    if np.any(valid_mask):
                        valid_indices = np.where(valid_mask)[0]
                        invalid_indices = np.where(~valid_mask)[0]
                        
                        # Interpolate
                        if len(valid_indices) > 1:
                            interpolated = np.interp(invalid_indices, valid_indices, 
                                                   signal[valid_indices])
                            signal[invalid_indices] = interpolated
                        else:
                            # If too few valid points, replace with zeros
                            signal[invalid_indices] = 0.0
                    else:
                        # If no valid points, replace entire signal with zeros
                        signal[:] = 0.0
                
                data[sample_idx, ch_idx, :] = signal
        
        return data
    
    def get_preprocessing_info(self) -> Dict:
        """Get information about preprocessing parameters."""
        return {
            'sampling_rate': self.sampling_rate,
            'lowpass_freq': self.lowpass_freq,
            'highpass_freq': self.highpass_freq,
            'notch_freq': self.notch_freq,
            'artifact_threshold': self.artifact_threshold,
            'baseline_method': self.baseline_method,
            'scaling_method': self.scaling_method,
            'is_fitted': self.is_fitted
        }


def create_optimal_preprocessor(sampling_rate: float = 128.0,
                              task_type: str = 'image_reconstruction') -> EEGPreprocessor:
    """
    Create optimally configured preprocessor for specific tasks.
    
    Args:
        sampling_rate: EEG sampling rate
        task_type: Type of task ('image_reconstruction', 'classification', 'general')
        
    Returns:
        EEGPreprocessor: Configured preprocessor
    """
    if task_type == 'image_reconstruction':
        # Preserve high-frequency information for image details
        return EEGPreprocessor(
            sampling_rate=sampling_rate,
            lowpass_freq=min(60.0, sampling_rate/2 - 5),  # Preserve high frequencies
            highpass_freq=0.1,  # Remove very slow drifts
            notch_freq=50.0 if sampling_rate > 100 else 0,  # Power line
            artifact_threshold=4.0,  # Moderate artifact removal
            baseline_method='detrend',
            scaling_method='robust'  # Robust to outliers
        )
    
    elif task_type == 'classification':
        # Standard preprocessing for classification
        return EEGPreprocessor(
            sampling_rate=sampling_rate,
            lowpass_freq=40.0,
            highpass_freq=0.5,
            notch_freq=50.0,
            artifact_threshold=3.0,
            baseline_method='mean',
            scaling_method='standard'
        )
    
    else:  # general
        # Conservative preprocessing
        return EEGPreprocessor(
            sampling_rate=sampling_rate,
            lowpass_freq=50.0,
            highpass_freq=0.5,
            notch_freq=50.0,
            artifact_threshold=5.0,
            baseline_method='detrend',
            scaling_method='robust'
        )
