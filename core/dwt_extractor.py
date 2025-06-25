#!/usr/bin/env python3
"""
Discrete Wavelet Transform (DWT) Feature Extraction
==================================================

This module implements comprehensive DWT-based feature extraction for EEG signals.
It provides multiple DWT analysis methods including multi-level decomposition,
statistical analysis, and frequency band characterization.

Author: EEG Research Team
Date: 2025-06-25
"""

import numpy as np
import pywt
from typing import List, Dict, Tuple, Optional
import logging

from .wavelet_base import WaveletFeatureBase, WaveletAnalyzer

logger = logging.getLogger(__name__)


class DWTFeatureExtractor(WaveletFeatureBase):
    """
    Comprehensive DWT feature extractor for EEG signals.
    
    This class implements multi-level DWT decomposition with extensive
    statistical feature extraction from approximation and detail coefficients.
    """
    
    def __init__(self, 
                 wavelet: str = 'db4',
                 levels: int = 4,
                 sampling_rate: float = 128.0,
                 include_approximation: bool = True,
                 include_details: bool = True,
                 feature_types: List[str] = None,
                 **kwargs):
        """
        Initialize DWT feature extractor.
        
        Args:
            wavelet (str): Wavelet type
            levels (int): Number of decomposition levels
            sampling_rate (float): Sampling rate in Hz
            include_approximation (bool): Include approximation coefficients
            include_details (bool): Include detail coefficients
            feature_types (List[str]): Types of features to extract
            **kwargs: Additional parameters
        """
        super().__init__("DWTFeatures", wavelet, levels, sampling_rate, **kwargs)
        
        self.include_approximation = include_approximation
        self.include_details = include_details
        
        # Default feature types
        if feature_types is None:
            self.feature_types = [
                'statistical', 'energy', 'entropy', 'frequency_analysis'
            ]
        else:
            self.feature_types = feature_types
        
        logger.info(f"DWT extractor configured with {len(self.feature_types)} feature types")
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract DWT features from EEG data.
        
        Args:
            data (np.ndarray): EEG data
            
        Returns:
            np.ndarray: Extracted DWT features
        """
        data = self.validate_data(data)
        data = self.reshape_data(data)
        
        n_samples, n_channels, n_timepoints = data.shape
        all_features = []
        
        logger.info(f"Extracting DWT features from {n_samples} samples, {n_channels} channels")
        
        for sample_idx in range(n_samples):
            sample_features = []
            
            for channel_idx in range(n_channels):
                signal_data = data[sample_idx, channel_idx, :]
                channel_features = self._extract_channel_dwt_features(signal_data)
                sample_features.extend(channel_features)
            
            all_features.append(sample_features)
        
        features_array = np.array(all_features)
        self.n_features = features_array.shape[1]
        
        # Generate feature names
        self._generate_dwt_feature_names(n_channels)
        
        logger.info(f"Extracted {self.n_features} DWT features")
        return features_array
    
    def _extract_channel_dwt_features(self, signal_data: np.ndarray) -> List[float]:
        """
        Extract DWT features from a single channel.
        
        Args:
            signal_data (np.ndarray): Single channel EEG signal
            
        Returns:
            List[float]: Extracted features
        """
        features = []
        
        # Perform DWT decomposition
        coeffs = pywt.wavedec(signal_data, self.wavelet, level=self.levels)
        
        # Extract features from each level
        for level_idx, coeff in enumerate(coeffs):
            if level_idx == 0 and not self.include_approximation:
                continue
            if level_idx > 0 and not self.include_details:
                continue
            
            level_features = self._extract_coefficient_features(coeff, level_idx)
            features.extend(level_features)
        
        # Add cross-level features
        if 'cross_level' in self.feature_types:
            cross_features = self._extract_cross_level_features(coeffs)
            features.extend(cross_features)
        
        return features
    
    def _extract_coefficient_features(self, coefficients: np.ndarray, level: int) -> List[float]:
        """
        Extract features from wavelet coefficients at a specific level.
        
        Args:
            coefficients (np.ndarray): Wavelet coefficients
            level (int): Decomposition level (0 = approximation, >0 = details)
            
        Returns:
            List[float]: Extracted features
        """
        features = []
        
        # Statistical features
        if 'statistical' in self.feature_types:
            stat_features = self.compute_statistical_features(coefficients)
            features.extend(stat_features)
        
        # Energy-based features
        if 'energy' in self.feature_types:
            energy_features = self._compute_energy_features(coefficients)
            features.extend(energy_features)
        
        # Entropy features
        if 'entropy' in self.feature_types:
            entropy_features = self._compute_entropy_features(coefficients)
            features.extend(entropy_features)
        
        # Frequency analysis features
        if 'frequency_analysis' in self.feature_types:
            freq_features = self._compute_frequency_features(coefficients, level)
            features.extend(freq_features)
        
        # Morphological features
        if 'morphological' in self.feature_types:
            morph_features = self._compute_morphological_features(coefficients)
            features.extend(morph_features)
        
        return features
    
    def _compute_energy_features(self, coefficients: np.ndarray) -> List[float]:
        """Compute energy-based features."""
        if len(coefficients) == 0:
            return [0.0] * 4
        
        # Total energy
        total_energy = np.sum(coefficients**2)
        
        # Mean energy
        mean_energy = total_energy / len(coefficients)
        
        # Energy concentration (ratio of max energy to total)
        max_energy = np.max(coefficients**2)
        energy_concentration = max_energy / total_energy if total_energy > 0 else 0.0
        
        # Energy distribution (std of squared coefficients)
        energy_std = np.std(coefficients**2)
        
        return [total_energy, mean_energy, energy_concentration, energy_std]
    
    def _compute_entropy_features(self, coefficients: np.ndarray) -> List[float]:
        """Compute entropy-based features."""
        if len(coefficients) == 0:
            return [0.0] * 3
        
        features = []
        
        # Shannon entropy
        coeff_abs = np.abs(coefficients)
        if np.sum(coeff_abs) > 0:
            prob = coeff_abs / np.sum(coeff_abs)
            prob = prob[prob > 0]
            shannon_entropy = -np.sum(prob * np.log2(prob))
        else:
            shannon_entropy = 0.0
        features.append(shannon_entropy)
        
        # Log energy entropy
        if np.sum(coefficients**2) > 0:
            energy_prob = coefficients**2 / np.sum(coefficients**2)
            energy_prob = energy_prob[energy_prob > 0]
            log_energy_entropy = -np.sum(energy_prob * np.log2(energy_prob))
        else:
            log_energy_entropy = 0.0
        features.append(log_energy_entropy)
        
        # Threshold entropy (entropy of thresholded coefficients)
        threshold = np.std(coefficients)
        thresholded = np.abs(coefficients) > threshold
        if np.sum(thresholded) > 0:
            p_above = np.sum(thresholded) / len(coefficients)
            p_below = 1 - p_above
            if p_above > 0 and p_below > 0:
                threshold_entropy = -(p_above * np.log2(p_above) + p_below * np.log2(p_below))
            else:
                threshold_entropy = 0.0
        else:
            threshold_entropy = 0.0
        features.append(threshold_entropy)
        
        return features
    
    def _compute_frequency_features(self, coefficients: np.ndarray, level: int) -> List[float]:
        """Compute frequency-related features."""
        if len(coefficients) == 0:
            return [0.0] * 3
        
        # Frequency band information
        nyquist = self.sampling_rate / 2
        
        if level == 0:  # Approximation
            freq_low = 0
            freq_high = nyquist / (2 ** self.levels)
        else:  # Detail
            actual_level = self.levels - level + 1
            freq_low = nyquist / (2 ** (actual_level + 1))
            freq_high = nyquist / (2 ** actual_level)
        
        # Center frequency
        center_freq = (freq_low + freq_high) / 2
        
        # Bandwidth
        bandwidth = freq_high - freq_low
        
        # Relative bandwidth
        rel_bandwidth = bandwidth / nyquist if nyquist > 0 else 0.0
        
        return [center_freq, bandwidth, rel_bandwidth]
    
    def _compute_morphological_features(self, coefficients: np.ndarray) -> List[float]:
        """Compute morphological features."""
        if len(coefficients) == 0:
            return [0.0] * 5
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(coefficients)) != 0)
        zcr = zero_crossings / len(coefficients) if len(coefficients) > 1 else 0.0
        
        # Peak count
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(np.abs(coefficients))
        peak_count = len(peaks)
        peak_density = peak_count / len(coefficients)
        
        # Coefficient sparsity (ratio of near-zero coefficients)
        threshold = 0.1 * np.std(coefficients)
        sparse_count = np.sum(np.abs(coefficients) < threshold)
        sparsity = sparse_count / len(coefficients)
        
        # Regularity measure (autocorrelation at lag 1)
        if len(coefficients) > 1:
            autocorr = np.corrcoef(coefficients[:-1], coefficients[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        
        return [zcr, peak_density, sparsity, autocorr, peak_count]
    
    def _extract_cross_level_features(self, coeffs: List[np.ndarray]) -> List[float]:
        """Extract features that compare across decomposition levels."""
        features = []
        
        if len(coeffs) < 2:
            return [0.0] * 4
        
        # Energy distribution across levels
        energies = [np.sum(c**2) for c in coeffs]
        total_energy = sum(energies)
        
        if total_energy > 0:
            energy_ratios = [e / total_energy for e in energies]
            
            # Energy concentration in approximation vs details
            approx_energy_ratio = energy_ratios[0]
            detail_energy_ratio = sum(energy_ratios[1:])
            
            # Energy distribution entropy
            energy_entropy = -sum(r * np.log2(r) for r in energy_ratios if r > 0)
            
            # Dominant level (level with maximum energy)
            dominant_level = np.argmax(energies)
            
            features.extend([approx_energy_ratio, detail_energy_ratio, 
                           energy_entropy, dominant_level])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def _generate_dwt_feature_names(self, n_channels: int):
        """Generate feature names for DWT features."""
        self.feature_names = []
        
        for channel in range(n_channels):
            channel_prefix = f"ch{channel:02d}"
            
            # Features for each level
            for level in range(self.levels + 1):
                if level == 0:
                    if not self.include_approximation:
                        continue
                    level_prefix = f"{channel_prefix}_dwt_A{self.levels}"
                else:
                    if not self.include_details:
                        continue
                    actual_level = self.levels - level + 1
                    level_prefix = f"{channel_prefix}_dwt_D{actual_level}"
                
                # Statistical features
                if 'statistical' in self.feature_types:
                    stat_names = self.get_statistical_feature_names(level_prefix)
                    self.feature_names.extend(stat_names)
                
                # Energy features
                if 'energy' in self.feature_types:
                    energy_names = [
                        f"{level_prefix}_total_energy",
                        f"{level_prefix}_mean_energy",
                        f"{level_prefix}_energy_concentration",
                        f"{level_prefix}_energy_std"
                    ]
                    self.feature_names.extend(energy_names)
                
                # Entropy features
                if 'entropy' in self.feature_types:
                    entropy_names = [
                        f"{level_prefix}_shannon_entropy",
                        f"{level_prefix}_log_energy_entropy",
                        f"{level_prefix}_threshold_entropy"
                    ]
                    self.feature_names.extend(entropy_names)
                
                # Frequency features
                if 'frequency_analysis' in self.feature_types:
                    freq_names = [
                        f"{level_prefix}_center_freq",
                        f"{level_prefix}_bandwidth",
                        f"{level_prefix}_rel_bandwidth"
                    ]
                    self.feature_names.extend(freq_names)
                
                # Morphological features
                if 'morphological' in self.feature_types:
                    morph_names = [
                        f"{level_prefix}_zcr",
                        f"{level_prefix}_peak_density",
                        f"{level_prefix}_sparsity",
                        f"{level_prefix}_autocorr",
                        f"{level_prefix}_peak_count"
                    ]
                    self.feature_names.extend(morph_names)
            
            # Cross-level features
            if 'cross_level' in self.feature_types:
                cross_names = [
                    f"{channel_prefix}_dwt_approx_energy_ratio",
                    f"{channel_prefix}_dwt_detail_energy_ratio",
                    f"{channel_prefix}_dwt_energy_entropy",
                    f"{channel_prefix}_dwt_dominant_level"
                ]
                self.feature_names.extend(cross_names)
    
    def analyze_decomposition(self, signal_data: np.ndarray) -> Dict:
        """
        Analyze DWT decomposition for a single signal.
        
        Args:
            signal_data (np.ndarray): Input signal
            
        Returns:
            Dict: Decomposition analysis
        """
        coeffs = pywt.wavedec(signal_data, self.wavelet, level=self.levels)
        
        analysis = {
            'wavelet': self.wavelet,
            'levels': self.levels,
            'signal_length': len(signal_data),
            'coefficients': {}
        }
        
        # Analyze each level
        for i, coeff in enumerate(coeffs):
            if i == 0:
                level_name = f"Approximation_L{self.levels}"
            else:
                level_name = f"Detail_L{self.levels - i + 1}"
            
            analysis['coefficients'][level_name] = {
                'length': len(coeff),
                'energy': np.sum(coeff**2),
                'mean': np.mean(coeff),
                'std': np.std(coeff),
                'max_abs': np.max(np.abs(coeff))
            }
        
        # Add frequency band analysis
        freq_analysis = WaveletAnalyzer.analyze_frequency_bands(coeffs, self.sampling_rate)
        analysis['frequency_bands'] = freq_analysis
        
        return analysis


class AdaptiveDWTExtractor(WaveletFeatureBase):
    """
    Adaptive DWT extractor that automatically selects optimal parameters.
    
    This extractor analyzes the input signal to determine the best wavelet
    and decomposition levels for feature extraction.
    """
    
    def __init__(self, 
                 candidate_wavelets: List[str] = None,
                 max_levels: int = 6,
                 sampling_rate: float = 128.0,
                 selection_criterion: str = 'energy_concentration',
                 **kwargs):
        """
        Initialize adaptive DWT extractor.
        
        Args:
            candidate_wavelets (List[str]): Wavelets to consider
            max_levels (int): Maximum decomposition levels
            sampling_rate (float): Sampling rate in Hz
            selection_criterion (str): Criterion for parameter selection
            **kwargs: Additional parameters
        """
        if candidate_wavelets is None:
            candidate_wavelets = WaveletAnalyzer.recommend_wavelet('eeg')
        
        super().__init__("AdaptiveDWT", candidate_wavelets[0], 4, sampling_rate, **kwargs)
        
        self.candidate_wavelets = candidate_wavelets
        self.max_levels = max_levels
        self.selection_criterion = selection_criterion
        self.selected_params = {}
        
        logger.info(f"Adaptive DWT with {len(candidate_wavelets)} candidate wavelets")
    
    def fit(self, data: np.ndarray) -> 'AdaptiveDWTExtractor':
        """
        Fit the extractor by selecting optimal parameters.
        
        Args:
            data (np.ndarray): Training data
            
        Returns:
            self: Fitted extractor
        """
        data = self.validate_data(data)
        data = self.reshape_data(data)
        
        logger.info("Selecting optimal DWT parameters...")
        
        # Use a subset of data for parameter selection
        n_samples = min(100, data.shape[0])
        sample_data = data[:n_samples]
        
        best_params = self._select_optimal_parameters(sample_data)
        
        self.wavelet = best_params['wavelet']
        self.levels = best_params['levels']
        self.selected_params = best_params
        
        logger.info(f"Selected wavelet: {self.wavelet}, levels: {self.levels}")
        
        self.is_fitted = True
        return self
    
    def _select_optimal_parameters(self, data: np.ndarray) -> Dict:
        """Select optimal wavelet and levels based on the data."""
        best_score = -np.inf
        best_params = {'wavelet': self.candidate_wavelets[0], 'levels': 4}
        
        for wavelet in self.candidate_wavelets:
            for levels in range(2, min(self.max_levels + 1, 7)):
                try:
                    score = self._evaluate_parameters(data, wavelet, levels)
                    if score > best_score:
                        best_score = score
                        best_params = {'wavelet': wavelet, 'levels': levels}
                except Exception as e:
                    logger.warning(f"Failed to evaluate {wavelet}, {levels}: {e}")
                    continue
        
        return best_params
    
    def _evaluate_parameters(self, data: np.ndarray, wavelet: str, levels: int) -> float:
        """Evaluate wavelet parameters based on selection criterion."""
        scores = []
        
        # Evaluate on multiple channels and samples
        n_samples, n_channels, n_timepoints = data.shape
        
        for sample_idx in range(min(10, n_samples)):
            for channel_idx in range(min(3, n_channels)):
                signal = data[sample_idx, channel_idx, :]
                
                try:
                    coeffs = pywt.wavedec(signal, wavelet, level=levels)
                    score = self._compute_criterion_score(coeffs)
                    scores.append(score)
                except:
                    continue
        
        return np.mean(scores) if scores else -np.inf
    
    def _compute_criterion_score(self, coeffs: List[np.ndarray]) -> float:
        """Compute score based on selection criterion."""
        if self.selection_criterion == 'energy_concentration':
            # Prefer decompositions with concentrated energy
            energies = [np.sum(c**2) for c in coeffs]
            total_energy = sum(energies)
            if total_energy > 0:
                energy_ratios = [e / total_energy for e in energies]
                # Higher concentration = higher score
                return max(energy_ratios)
            return 0.0
        
        elif self.selection_criterion == 'entropy':
            # Prefer decompositions with balanced energy distribution
            energies = [np.sum(c**2) for c in coeffs]
            total_energy = sum(energies)
            if total_energy > 0:
                energy_ratios = [e / total_energy for e in energies if e > 0]
                return -sum(r * np.log2(r) for r in energy_ratios)
            return 0.0
        
        else:
            return 0.0
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features using selected parameters."""
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before extracting features")
        
        # Use the base DWT extractor with selected parameters
        dwt_extractor = DWTFeatureExtractor(
            wavelet=self.wavelet,
            levels=self.levels,
            sampling_rate=self.sampling_rate
        )
        
        return dwt_extractor.extract_features(data)
