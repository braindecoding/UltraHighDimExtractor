"""
Quality Metrics and Evaluation
==============================

This module provides comprehensive metrics for evaluating feature quality
and reconstruction performance.

Classes:
- FeatureQualityMetrics: Metrics for feature quality assessment
- ReconstructionMetrics: Metrics for image reconstruction evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
import logging

logger = logging.getLogger(__name__)


class FeatureQualityMetrics:
    """
    Comprehensive feature quality assessment.
    
    This class provides various metrics to evaluate the quality
    and informativeness of extracted features.
    """
    
    @staticmethod
    def compute_basic_metrics(features: np.ndarray) -> Dict[str, float]:
        """
        Compute basic feature quality metrics.
        
        Args:
            features: Feature array (n_samples, n_features)
            
        Returns:
            Dict[str, float]: Basic metrics
        """
        n_samples, n_features = features.shape
        
        # Basic statistics
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        feature_vars = np.var(features, axis=0)
        
        metrics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'mean_value': float(np.mean(features)),
            'std_value': float(np.std(features)),
            'min_value': float(np.min(features)),
            'max_value': float(np.max(features)),
            'n_nan': int(np.sum(np.isnan(features))),
            'n_inf': int(np.sum(np.isinf(features))),
            'n_zero_variance': int(np.sum(feature_vars == 0)),
            'mean_feature_std': float(np.mean(feature_stds)),
            'std_feature_std': float(np.std(feature_stds)),
            'feature_std_range': [float(np.min(feature_stds)), float(np.max(feature_stds))]
        }
        
        return metrics
    
    @staticmethod
    def compute_correlation_metrics(features: np.ndarray) -> Dict[str, float]:
        """
        Compute feature correlation metrics.
        
        Args:
            features: Feature array
            
        Returns:
            Dict[str, float]: Correlation metrics
        """
        # Compute correlation matrix
        corr_matrix = np.corrcoef(features.T)
        
        # Remove diagonal (self-correlations)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        off_diagonal_corrs = corr_matrix[mask]
        
        # Remove NaN values
        valid_corrs = off_diagonal_corrs[~np.isnan(off_diagonal_corrs)]
        
        metrics = {
            'mean_correlation': float(np.mean(valid_corrs)) if len(valid_corrs) > 0 else 0.0,
            'std_correlation': float(np.std(valid_corrs)) if len(valid_corrs) > 0 else 0.0,
            'max_correlation': float(np.max(valid_corrs)) if len(valid_corrs) > 0 else 0.0,
            'min_correlation': float(np.min(valid_corrs)) if len(valid_corrs) > 0 else 0.0,
            'high_correlation_pairs': int(np.sum(np.abs(valid_corrs) > 0.9)) if len(valid_corrs) > 0 else 0
        }
        
        return metrics
    
    @staticmethod
    def compute_information_metrics(features: np.ndarray, 
                                  target: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute information-theoretic metrics.
        
        Args:
            features: Feature array
            target: Target variable (optional)
            
        Returns:
            Dict[str, float]: Information metrics
        """
        metrics = {}
        
        # Feature variance distribution
        feature_vars = np.var(features, axis=0)
        metrics['variance_entropy'] = float(-np.sum(feature_vars * np.log(feature_vars + 1e-10)))
        metrics['effective_features'] = int(np.sum(feature_vars > 0.01))
        
        # Mutual information with target (if provided)
        if target is not None:
            try:
                # Sample features for computational efficiency
                n_features = features.shape[1]
                if n_features > 1000:
                    sample_indices = np.random.choice(n_features, 1000, replace=False)
                    sample_features = features[:, sample_indices]
                else:
                    sample_features = features
                
                mi_scores = mutual_info_regression(sample_features, target)
                metrics['mean_mutual_info'] = float(np.mean(mi_scores))
                metrics['max_mutual_info'] = float(np.max(mi_scores))
                metrics['informative_features'] = int(np.sum(mi_scores > 0.1))
                
            except Exception as e:
                logger.warning(f"Mutual information computation failed: {e}")
                metrics['mean_mutual_info'] = 0.0
                metrics['max_mutual_info'] = 0.0
                metrics['informative_features'] = 0
        
        return metrics
    
    @staticmethod
    def compute_stability_metrics(features_1: np.ndarray, 
                                features_2: np.ndarray) -> Dict[str, float]:
        """
        Compute feature stability metrics between two feature sets.
        
        Args:
            features_1: First feature set
            features_2: Second feature set
            
        Returns:
            Dict[str, float]: Stability metrics
        """
        if features_1.shape != features_2.shape:
            raise ValueError("Feature sets must have same shape")
        
        # Feature-wise correlations
        feature_correlations = []
        for i in range(features_1.shape[1]):
            corr = np.corrcoef(features_1[:, i], features_2[:, i])[0, 1]
            if not np.isnan(corr):
                feature_correlations.append(corr)
        
        # Overall similarity
        overall_corr = np.corrcoef(features_1.flatten(), features_2.flatten())[0, 1]
        
        metrics = {
            'mean_feature_correlation': float(np.mean(feature_correlations)) if feature_correlations else 0.0,
            'std_feature_correlation': float(np.std(feature_correlations)) if feature_correlations else 0.0,
            'stable_features': int(np.sum(np.array(feature_correlations) > 0.8)) if feature_correlations else 0,
            'overall_correlation': float(overall_corr) if not np.isnan(overall_corr) else 0.0
        }
        
        return metrics
    
    @classmethod
    def comprehensive_assessment(cls, features: np.ndarray, 
                               target: Optional[np.ndarray] = None) -> Dict:
        """
        Perform comprehensive feature quality assessment.
        
        Args:
            features: Feature array
            target: Target variable (optional)
            
        Returns:
            Dict: Comprehensive assessment report
        """
        report = {
            'basic_metrics': cls.compute_basic_metrics(features),
            'correlation_metrics': cls.compute_correlation_metrics(features),
            'information_metrics': cls.compute_information_metrics(features, target)
        }
        
        # Overall quality score
        basic = report['basic_metrics']
        corr = report['correlation_metrics']
        info = report['information_metrics']
        
        # Compute quality score (0-100)
        quality_score = 0
        
        # Penalize missing/invalid values
        if basic['n_nan'] == 0 and basic['n_inf'] == 0:
            quality_score += 20
        
        # Reward high dimensionality
        if basic['n_features'] > 10000:
            quality_score += 20
        elif basic['n_features'] > 5000:
            quality_score += 15
        elif basic['n_features'] > 1000:
            quality_score += 10
        
        # Reward low redundancy
        if corr['mean_correlation'] < 0.3:
            quality_score += 20
        elif corr['mean_correlation'] < 0.5:
            quality_score += 15
        elif corr['mean_correlation'] < 0.7:
            quality_score += 10
        
        # Reward informative features
        effective_ratio = info['effective_features'] / basic['n_features']
        if effective_ratio > 0.8:
            quality_score += 20
        elif effective_ratio > 0.6:
            quality_score += 15
        elif effective_ratio > 0.4:
            quality_score += 10
        
        # Reward target informativeness (if available)
        if target is not None and info['mean_mutual_info'] > 0:
            if info['mean_mutual_info'] > 0.5:
                quality_score += 20
            elif info['mean_mutual_info'] > 0.3:
                quality_score += 15
            elif info['mean_mutual_info'] > 0.1:
                quality_score += 10
        
        report['quality_score'] = min(100, quality_score)
        
        return report


class ReconstructionMetrics:
    """
    Metrics for evaluating image reconstruction quality.
    
    This class provides various metrics to assess the quality
    of reconstructed images compared to ground truth.
    """
    
    @staticmethod
    def compute_pixel_metrics(true_images: np.ndarray, 
                            reconstructed_images: np.ndarray) -> Dict[str, float]:
        """
        Compute pixel-level reconstruction metrics.
        
        Args:
            true_images: Ground truth images
            reconstructed_images: Reconstructed images
            
        Returns:
            Dict[str, float]: Pixel-level metrics
        """
        # Flatten images
        true_flat = true_images.reshape(true_images.shape[0], -1)
        recon_flat = reconstructed_images.reshape(reconstructed_images.shape[0], -1)
        
        # Basic metrics
        mse = mean_squared_error(true_flat, recon_flat)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_flat - recon_flat))
        r2 = r2_score(true_flat, recon_flat)
        
        # Normalized metrics
        true_range = np.max(true_flat) - np.min(true_flat)
        normalized_rmse = rmse / true_range if true_range > 0 else float('inf')
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'normalized_rmse': float(normalized_rmse)
        }
        
        return metrics
    
    @staticmethod
    def compute_correlation_metrics(true_images: np.ndarray,
                                  reconstructed_images: np.ndarray) -> Dict[str, float]:
        """
        Compute correlation-based reconstruction metrics.
        
        Args:
            true_images: Ground truth images
            reconstructed_images: Reconstructed images
            
        Returns:
            Dict[str, float]: Correlation metrics
        """
        # Flatten images
        true_flat = true_images.reshape(true_images.shape[0], -1)
        recon_flat = reconstructed_images.reshape(reconstructed_images.shape[0], -1)
        
        # Per-image correlations
        correlations = []
        for i in range(len(true_flat)):
            corr = np.corrcoef(true_flat[i], recon_flat[i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        # Overall correlation
        overall_corr = np.corrcoef(true_flat.flatten(), recon_flat.flatten())[0, 1]
        
        metrics = {
            'mean_correlation': float(np.mean(correlations)) if correlations else 0.0,
            'std_correlation': float(np.std(correlations)) if correlations else 0.0,
            'min_correlation': float(np.min(correlations)) if correlations else 0.0,
            'max_correlation': float(np.max(correlations)) if correlations else 0.0,
            'overall_correlation': float(overall_corr) if not np.isnan(overall_corr) else 0.0,
            'good_reconstructions': int(np.sum(np.array(correlations) > 0.5)) if correlations else 0
        }
        
        return metrics
    
    @staticmethod
    def compute_structural_metrics(true_images: np.ndarray,
                                 reconstructed_images: np.ndarray) -> Dict[str, float]:
        """
        Compute structural similarity metrics.
        
        Args:
            true_images: Ground truth images
            reconstructed_images: Reconstructed images
            
        Returns:
            Dict[str, float]: Structural metrics
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            
            ssim_scores = []
            for i in range(len(true_images)):
                # Reshape to 2D if needed
                if true_images.ndim == 2:
                    true_img = true_images[i].reshape(int(np.sqrt(true_images.shape[1])), -1)
                    recon_img = reconstructed_images[i].reshape(int(np.sqrt(reconstructed_images.shape[1])), -1)
                else:
                    true_img = true_images[i]
                    recon_img = reconstructed_images[i]
                
                score = ssim(true_img, recon_img, data_range=true_img.max() - true_img.min())
                ssim_scores.append(score)
            
            metrics = {
                'mean_ssim': float(np.mean(ssim_scores)),
                'std_ssim': float(np.std(ssim_scores)),
                'min_ssim': float(np.min(ssim_scores)),
                'max_ssim': float(np.max(ssim_scores))
            }
            
        except ImportError:
            logger.warning("scikit-image not available, skipping SSIM computation")
            metrics = {
                'mean_ssim': 0.0,
                'std_ssim': 0.0,
                'min_ssim': 0.0,
                'max_ssim': 0.0
            }
        
        return metrics
    
    @classmethod
    def comprehensive_evaluation(cls, true_images: np.ndarray,
                               reconstructed_images: np.ndarray) -> Dict:
        """
        Perform comprehensive reconstruction evaluation.
        
        Args:
            true_images: Ground truth images
            reconstructed_images: Reconstructed images
            
        Returns:
            Dict: Comprehensive evaluation report
        """
        if true_images.shape != reconstructed_images.shape:
            raise ValueError("True and reconstructed images must have same shape")
        
        report = {
            'pixel_metrics': cls.compute_pixel_metrics(true_images, reconstructed_images),
            'correlation_metrics': cls.compute_correlation_metrics(true_images, reconstructed_images),
            'structural_metrics': cls.compute_structural_metrics(true_images, reconstructed_images)
        }
        
        # Overall reconstruction score (0-100)
        pixel = report['pixel_metrics']
        corr = report['correlation_metrics']
        struct = report['structural_metrics']
        
        reconstruction_score = 0
        
        # R² score contribution (0-30 points)
        if pixel['r2_score'] > 0.8:
            reconstruction_score += 30
        elif pixel['r2_score'] > 0.6:
            reconstruction_score += 25
        elif pixel['r2_score'] > 0.4:
            reconstruction_score += 20
        elif pixel['r2_score'] > 0.2:
            reconstruction_score += 15
        elif pixel['r2_score'] > 0:
            reconstruction_score += 10
        
        # Correlation contribution (0-35 points)
        if corr['mean_correlation'] > 0.8:
            reconstruction_score += 35
        elif corr['mean_correlation'] > 0.6:
            reconstruction_score += 30
        elif corr['mean_correlation'] > 0.4:
            reconstruction_score += 25
        elif corr['mean_correlation'] > 0.2:
            reconstruction_score += 20
        elif corr['mean_correlation'] > 0:
            reconstruction_score += 15
        
        # SSIM contribution (0-35 points)
        if struct['mean_ssim'] > 0.8:
            reconstruction_score += 35
        elif struct['mean_ssim'] > 0.6:
            reconstruction_score += 30
        elif struct['mean_ssim'] > 0.4:
            reconstruction_score += 25
        elif struct['mean_ssim'] > 0.2:
            reconstruction_score += 20
        elif struct['mean_ssim'] > 0:
            reconstruction_score += 15
        
        report['reconstruction_score'] = min(100, reconstruction_score)
        
        return report
