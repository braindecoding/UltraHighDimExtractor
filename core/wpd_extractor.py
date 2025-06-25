#!/usr/bin/env python3
"""
Wavelet Packet Decomposition (WPD) Feature Extraction
====================================================

This module implements comprehensive WPD-based feature extraction for EEG signals.
WPD provides a more detailed frequency analysis compared to DWT by decomposing
both approximation and detail coefficients at each level.

Author: EEG Research Team
Date: 2025-06-25
"""

import numpy as np
import pywt
from typing import List, Dict, Tuple, Optional, Set
import logging

from wavelet_base import WaveletFeatureBase

logger = logging.getLogger(__name__)


class WPDFeatureExtractor(WaveletFeatureBase):
    """
    Comprehensive WPD feature extractor for EEG signals.
    
    This class implements wavelet packet decomposition with extensive
    feature extraction from all packet nodes, providing detailed
    frequency-domain analysis.
    """
    
    def __init__(self, 
                 wavelet: str = 'db4',
                 levels: int = 4,
                 sampling_rate: float = 128.0,
                 node_selection: str = 'all',
                 feature_types: List[str] = None,
                 energy_threshold: float = 0.01,
                 **kwargs):
        """
        Initialize WPD feature extractor.
        
        Args:
            wavelet (str): Wavelet type
            levels (int): Number of decomposition levels
            sampling_rate (float): Sampling rate in Hz
            node_selection (str): Node selection strategy ('all', 'best_basis', 'energy_threshold')
            feature_types (List[str]): Types of features to extract
            energy_threshold (float): Threshold for energy-based node selection
            **kwargs: Additional parameters
        """
        super().__init__("WPDFeatures", wavelet, levels, sampling_rate, **kwargs)
        
        self.node_selection = node_selection
        self.energy_threshold = energy_threshold
        
        # Default feature types
        if feature_types is None:
            self.feature_types = ['statistical', 'energy', 'frequency']
        else:
            self.feature_types = feature_types
        
        # Track selected nodes
        self.selected_nodes = []
        
        logger.info(f"WPD extractor configured with {node_selection} node selection")
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract WPD features from EEG data.
        
        Args:
            data (np.ndarray): EEG data
            
        Returns:
            np.ndarray: Extracted WPD features
        """
        data = self.validate_data(data)
        data = self.reshape_data(data)
        
        n_samples, n_channels, n_timepoints = data.shape
        all_features = []
        
        logger.info(f"Extracting WPD features from {n_samples} samples, {n_channels} channels")
        
        # Determine nodes to use (use first sample for node selection)
        if not self.selected_nodes:
            sample_signal = data[0, 0, :]
            self.selected_nodes = self._select_nodes(sample_signal)
            logger.info(f"Selected {len(self.selected_nodes)} WPD nodes")
        
        for sample_idx in range(n_samples):
            sample_features = []
            
            for channel_idx in range(n_channels):
                signal_data = data[sample_idx, channel_idx, :]
                channel_features = self._extract_channel_wpd_features(signal_data)
                sample_features.extend(channel_features)
            
            all_features.append(sample_features)
        
        features_array = np.array(all_features)
        self.n_features = features_array.shape[1]
        
        # Generate feature names
        self._generate_wpd_feature_names(n_channels)
        
        logger.info(f"Extracted {self.n_features} WPD features")
        return features_array
    
    def _select_nodes(self, signal_data: np.ndarray) -> List[str]:
        """
        Select WPD nodes based on the specified strategy.
        
        Args:
            signal_data (np.ndarray): Sample signal for node selection
            
        Returns:
            List[str]: Selected node paths
        """
        # Create wavelet packet tree
        wp = pywt.WaveletPacket(signal_data, self.wavelet, maxlevel=self.levels)
        
        if self.node_selection == 'all':
            # Use all nodes at the maximum level
            nodes = [node.path for node in wp.get_level(self.levels, 'freq')]
        
        elif self.node_selection == 'best_basis':
            # Use best basis decomposition
            best_tree = wp.get_level(self.levels, 'freq')
            nodes = [node.path for node in best_tree]
        
        elif self.node_selection == 'energy_threshold':
            # Select nodes based on energy threshold
            nodes = self._select_nodes_by_energy(wp, signal_data)
        
        else:
            # Default to all nodes at max level
            nodes = [node.path for node in wp.get_level(self.levels, 'freq')]
        
        return sorted(nodes)
    
    def _select_nodes_by_energy(self, wp: pywt.WaveletPacket, signal_data: np.ndarray) -> List[str]:
        """Select nodes based on energy threshold."""
        selected_nodes = []
        total_energy = np.sum(signal_data**2)
        
        # Check all nodes at maximum level
        for node in wp.get_level(self.levels, 'freq'):
            node_energy = np.sum(node.data**2)
            energy_ratio = node_energy / total_energy if total_energy > 0 else 0
            
            if energy_ratio >= self.energy_threshold:
                selected_nodes.append(node.path)
        
        # Ensure we have at least some nodes
        if not selected_nodes:
            # Fall back to all nodes if none meet threshold
            selected_nodes = [node.path for node in wp.get_level(self.levels, 'freq')]
        
        return selected_nodes
    
    def _extract_channel_wpd_features(self, signal_data: np.ndarray) -> List[float]:
        """
        Extract WPD features from a single channel.
        
        Args:
            signal_data (np.ndarray): Single channel EEG signal
            
        Returns:
            List[float]: Extracted features
        """
        features = []
        
        # Create wavelet packet tree
        wp = pywt.WaveletPacket(signal_data, self.wavelet, maxlevel=self.levels)
        
        # Extract features from selected nodes
        for node_path in self.selected_nodes:
            try:
                node = wp[node_path]
                node_features = self._extract_node_features(node.data, node_path)
                features.extend(node_features)
            except Exception as e:
                logger.warning(f"Failed to extract features from node {node_path}: {e}")
                # Add zeros for missing features
                n_features_per_node = self._get_features_per_node()
                features.extend([0.0] * n_features_per_node)
        
        # Add global WPD features
        global_features = self._extract_global_wpd_features(wp, signal_data)
        features.extend(global_features)
        
        return features
    
    def _extract_node_features(self, node_data: np.ndarray, node_path: str) -> List[float]:
        """
        Extract features from a single WPD node.
        
        Args:
            node_data (np.ndarray): Node coefficients
            node_path (str): Node path identifier
            
        Returns:
            List[float]: Node features
        """
        features = []
        
        if len(node_data) == 0:
            return [0.0] * self._get_features_per_node()
        
        # Statistical features
        if 'statistical' in self.feature_types:
            stat_features = self.compute_statistical_features(node_data)
            features.extend(stat_features)
        
        # Energy features
        if 'energy' in self.feature_types:
            energy_features = self._compute_node_energy_features(node_data)
            features.extend(energy_features)
        
        # Frequency features
        if 'frequency' in self.feature_types:
            freq_features = self._compute_node_frequency_features(node_path)
            features.extend(freq_features)
        
        # Spectral features
        if 'spectral' in self.feature_types:
            spectral_features = self._compute_node_spectral_features(node_data)
            features.extend(spectral_features)
        
        return features
    
    def _compute_node_energy_features(self, node_data: np.ndarray) -> List[float]:
        """Compute energy features for a node."""
        # Total energy
        total_energy = np.sum(node_data**2)
        
        # Normalized energy
        norm_energy = total_energy / len(node_data) if len(node_data) > 0 else 0.0
        
        # Energy concentration
        if total_energy > 0:
            max_coeff_energy = np.max(node_data**2)
            energy_concentration = max_coeff_energy / total_energy
        else:
            energy_concentration = 0.0
        
        return [total_energy, norm_energy, energy_concentration]
    
    def _compute_node_frequency_features(self, node_path: str) -> List[float]:
        """Compute frequency features based on node position."""
        # Decode frequency band from node path
        level = len(node_path)
        if level == 0:
            return [0.0, 0.0, 0.0]

        # Convert wavelet packet path to frequency band index
        # WPD paths use 'a' and 'd' characters, not binary digits
        try:
            # Convert 'a' to 0 and 'd' to 1, then interpret as binary
            binary_str = node_path.replace('a', '0').replace('d', '1')
            path_value = int(binary_str, 2) if binary_str else 0
        except ValueError:
            # Fallback: use simple enumeration based on path
            path_value = hash(node_path) % (2 ** level)

        total_bands = 2 ** level

        # Calculate frequency range
        nyquist = self.sampling_rate / 2
        band_width = nyquist / total_bands
        freq_low = path_value * band_width
        freq_high = (path_value + 1) * band_width
        center_freq = (freq_low + freq_high) / 2

        return [freq_low, freq_high, center_freq]
    
    def _compute_node_spectral_features(self, node_data: np.ndarray) -> List[float]:
        """Compute spectral features for a node."""
        if len(node_data) < 4:
            return [0.0, 0.0, 0.0]
        
        # Compute power spectral density
        from scipy import signal as scipy_signal
        
        try:
            freqs, psd = scipy_signal.welch(node_data, 
                                          fs=self.sampling_rate,
                                          nperseg=min(len(node_data), 64))
            
            # Spectral centroid
            if np.sum(psd) > 0:
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            else:
                spectral_centroid = 0.0
            
            # Spectral spread
            if np.sum(psd) > 0:
                spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
            else:
                spectral_spread = 0.0
            
            # Spectral rolloff (95% of energy)
            cumulative_psd = np.cumsum(psd)
            total_energy = cumulative_psd[-1]
            if total_energy > 0:
                rolloff_idx = np.where(cumulative_psd >= 0.95 * total_energy)[0]
                spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            else:
                spectral_rolloff = 0.0
            
            return [spectral_centroid, spectral_spread, spectral_rolloff]
            
        except Exception as e:
            logger.warning(f"Failed to compute spectral features: {e}")
            return [0.0, 0.0, 0.0]
    
    def _extract_global_wpd_features(self, wp: pywt.WaveletPacket, signal_data: np.ndarray) -> List[float]:
        """Extract global features from the entire WPD tree."""
        features = []
        
        # Energy distribution across selected nodes
        node_energies = []
        for node_path in self.selected_nodes:
            try:
                node = wp[node_path]
                energy = np.sum(node.data**2)
                node_energies.append(energy)
            except:
                node_energies.append(0.0)
        
        total_energy = sum(node_energies)
        
        if total_energy > 0:
            # Energy distribution entropy
            energy_probs = [e / total_energy for e in node_energies if e > 0]
            if energy_probs:
                energy_entropy = -sum(p * np.log2(p) for p in energy_probs)
            else:
                energy_entropy = 0.0
            
            # Energy concentration (max energy ratio)
            max_energy_ratio = max(node_energies) / total_energy
            
            # Number of significant nodes (energy > 5% of total)
            significant_nodes = sum(1 for e in node_energies if e / total_energy > 0.05)
            
        else:
            energy_entropy = 0.0
            max_energy_ratio = 0.0
            significant_nodes = 0
        
        features.extend([energy_entropy, max_energy_ratio, significant_nodes])
        
        # Tree depth utilization
        max_depth = self.levels
        avg_node_depth = np.mean([len(path) for path in self.selected_nodes])
        depth_utilization = avg_node_depth / max_depth if max_depth > 0 else 0.0
        
        features.append(depth_utilization)
        
        return features
    
    def _get_features_per_node(self) -> int:
        """Get number of features extracted per node."""
        count = 0
        if 'statistical' in self.feature_types:
            count += 12  # From compute_statistical_features
        if 'energy' in self.feature_types:
            count += 3   # From _compute_node_energy_features
        if 'frequency' in self.feature_types:
            count += 3   # From _compute_node_frequency_features
        if 'spectral' in self.feature_types:
            count += 3   # From _compute_node_spectral_features
        return count
    
    def _generate_wpd_feature_names(self, n_channels: int):
        """Generate feature names for WPD features."""
        self.feature_names = []
        
        for channel in range(n_channels):
            channel_prefix = f"ch{channel:02d}"
            
            # Features for each selected node
            for node_path in self.selected_nodes:
                node_prefix = f"{channel_prefix}_wpd_{node_path}"
                
                # Statistical features
                if 'statistical' in self.feature_types:
                    stat_names = self.get_statistical_feature_names(node_prefix)
                    self.feature_names.extend(stat_names)
                
                # Energy features
                if 'energy' in self.feature_types:
                    energy_names = [
                        f"{node_prefix}_total_energy",
                        f"{node_prefix}_norm_energy",
                        f"{node_prefix}_energy_concentration"
                    ]
                    self.feature_names.extend(energy_names)
                
                # Frequency features
                if 'frequency' in self.feature_types:
                    freq_names = [
                        f"{node_prefix}_freq_low",
                        f"{node_prefix}_freq_high",
                        f"{node_prefix}_center_freq"
                    ]
                    self.feature_names.extend(freq_names)
                
                # Spectral features
                if 'spectral' in self.feature_types:
                    spectral_names = [
                        f"{node_prefix}_spectral_centroid",
                        f"{node_prefix}_spectral_spread",
                        f"{node_prefix}_spectral_rolloff"
                    ]
                    self.feature_names.extend(spectral_names)
            
            # Global WPD features
            global_names = [
                f"{channel_prefix}_wpd_energy_entropy",
                f"{channel_prefix}_wpd_max_energy_ratio",
                f"{channel_prefix}_wpd_significant_nodes",
                f"{channel_prefix}_wpd_depth_utilization"
            ]
            self.feature_names.extend(global_names)
    
    def analyze_wpd_tree(self, signal_data: np.ndarray) -> Dict:
        """
        Analyze WPD tree structure and energy distribution.
        
        Args:
            signal_data (np.ndarray): Input signal
            
        Returns:
            Dict: WPD tree analysis
        """
        wp = pywt.WaveletPacket(signal_data, self.wavelet, maxlevel=self.levels)
        
        analysis = {
            'wavelet': self.wavelet,
            'levels': self.levels,
            'signal_length': len(signal_data),
            'total_nodes': 2 ** (self.levels + 1) - 1,
            'selected_nodes': len(self.selected_nodes),
            'nodes': {}
        }
        
        # Analyze each selected node
        for node_path in self.selected_nodes:
            try:
                node = wp[node_path]
                freq_features = self._compute_node_frequency_features(node_path)
                
                analysis['nodes'][node_path] = {
                    'level': len(node_path),
                    'length': len(node.data),
                    'energy': np.sum(node.data**2),
                    'freq_low': freq_features[0],
                    'freq_high': freq_features[1],
                    'center_freq': freq_features[2]
                }
            except Exception as e:
                logger.warning(f"Failed to analyze node {node_path}: {e}")
        
        return analysis


class OptimizedWPDExtractor(WaveletFeatureBase):
    """
    Optimized WPD extractor that selects the most informative nodes.
    
    This extractor uses various criteria to select a subset of WPD nodes
    that provide the most discriminative information for classification.
    """
    
    def __init__(self, 
                 wavelet: str = 'db4',
                 levels: int = 4,
                 sampling_rate: float = 128.0,
                 max_nodes: int = 16,
                 selection_method: str = 'energy_variance',
                 **kwargs):
        """
        Initialize optimized WPD extractor.
        
        Args:
            wavelet (str): Wavelet type
            levels (int): Number of decomposition levels
            sampling_rate (float): Sampling rate in Hz
            max_nodes (int): Maximum number of nodes to select
            selection_method (str): Node selection method
            **kwargs: Additional parameters
        """
        super().__init__("OptimizedWPD", wavelet, levels, sampling_rate, **kwargs)
        
        self.max_nodes = max_nodes
        self.selection_method = selection_method
        self.selected_nodes = []
        
        logger.info(f"Optimized WPD with max {max_nodes} nodes using {selection_method}")
    
    def fit(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> 'OptimizedWPDExtractor':
        """
        Fit the extractor by selecting optimal nodes.
        
        Args:
            data (np.ndarray): Training data
            labels (np.ndarray): Training labels (optional)
            
        Returns:
            self: Fitted extractor
        """
        data = self.validate_data(data)
        data = self.reshape_data(data)
        
        logger.info("Selecting optimal WPD nodes...")
        
        # Use subset of data for node selection
        n_samples = min(100, data.shape[0])
        sample_data = data[:n_samples]
        sample_labels = labels[:n_samples] if labels is not None else None
        
        self.selected_nodes = self._select_optimal_nodes(sample_data, sample_labels)
        
        logger.info(f"Selected {len(self.selected_nodes)} optimal nodes")
        
        self.is_fitted = True
        return self
    
    def _select_optimal_nodes(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> List[str]:
        """Select optimal nodes based on the selection method."""
        # Get all possible nodes at max level
        sample_signal = data[0, 0, :]
        wp = pywt.WaveletPacket(sample_signal, self.wavelet, maxlevel=self.levels)
        all_nodes = [node.path for node in wp.get_level(self.levels, 'freq')]
        
        if self.selection_method == 'energy_variance':
            return self._select_by_energy_variance(data, all_nodes)
        elif self.selection_method == 'discriminative' and labels is not None:
            return self._select_by_discriminative_power(data, labels, all_nodes)
        else:
            # Default: select by energy
            return self._select_by_energy(data, all_nodes)
    
    def _select_by_energy_variance(self, data: np.ndarray, all_nodes: List[str]) -> List[str]:
        """Select nodes with highest energy variance across samples."""
        node_scores = {}
        
        for node_path in all_nodes:
            energies = []
            
            # Compute energy for this node across samples
            for sample_idx in range(data.shape[0]):
                for channel_idx in range(min(3, data.shape[1])):  # Use first 3 channels
                    signal = data[sample_idx, channel_idx, :]
                    wp = pywt.WaveletPacket(signal, self.wavelet, maxlevel=self.levels)
                    try:
                        node = wp[node_path]
                        energy = np.sum(node.data**2)
                        energies.append(energy)
                    except:
                        continue
            
            # Score based on variance of energies
            if energies:
                node_scores[node_path] = np.var(energies)
            else:
                node_scores[node_path] = 0.0
        
        # Select top nodes
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [node for node, score in sorted_nodes[:self.max_nodes]]
        
        return selected
    
    def _select_by_energy(self, data: np.ndarray, all_nodes: List[str]) -> List[str]:
        """Select nodes with highest average energy."""
        node_scores = {}
        
        for node_path in all_nodes:
            energies = []
            
            for sample_idx in range(min(20, data.shape[0])):
                for channel_idx in range(min(3, data.shape[1])):
                    signal = data[sample_idx, channel_idx, :]
                    wp = pywt.WaveletPacket(signal, self.wavelet, maxlevel=self.levels)
                    try:
                        node = wp[node_path]
                        energy = np.sum(node.data**2)
                        energies.append(energy)
                    except:
                        continue
            
            node_scores[node_path] = np.mean(energies) if energies else 0.0
        
        # Select top nodes
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [node for node, score in sorted_nodes[:self.max_nodes]]
        
        return selected
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features using selected nodes."""
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before extracting features")
        
        # Use WPD extractor with selected nodes
        wpd_extractor = WPDFeatureExtractor(
            wavelet=self.wavelet,
            levels=self.levels,
            sampling_rate=self.sampling_rate
        )
        wpd_extractor.selected_nodes = self.selected_nodes
        
        return wpd_extractor.extract_features(data)
