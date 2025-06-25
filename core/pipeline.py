"""
Image Reconstruction Pipeline
============================

This module provides a complete pipeline for EEG-to-image reconstruction
using ultra-high dimensional wavelet features.

Classes:
- ImageReconstructionPipeline: Complete end-to-end pipeline
- FeatureSelectionPipeline: Feature selection and optimization
- ReconstructionModel: Image reconstruction model wrapper
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
import pickle

from .preprocessing import EEGPreprocessor, create_optimal_preprocessor
from .ultra_extractor import UltraHighDimExtractor
from ..utils.validation import validate_eeg_data
from ..utils.metrics import FeatureQualityMetrics

logger = logging.getLogger(__name__)


class ImageReconstructionPipeline:
    """
    Complete pipeline for EEG-to-image reconstruction.
    
    This pipeline combines preprocessing, feature extraction, feature selection,
    and image reconstruction in a unified interface.
    """
    
    def __init__(self,
                 target_dimensions: int = 40000,
                 preprocessing_config: Dict = None,
                 feature_selection: bool = True,
                 model_type: str = 'mlp',
                 **kwargs):
        """
        Initialize image reconstruction pipeline.
        
        Args:
            target_dimensions: Target number of features
            preprocessing_config: Preprocessing configuration
            feature_selection: Whether to apply feature selection
            model_type: Type of reconstruction model
        """
        self.target_dimensions = target_dimensions
        self.feature_selection = feature_selection
        self.model_type = model_type
        
        # Initialize components
        self.preprocessor = None
        self.feature_extractor = None
        self.feature_selector = None
        self.reconstruction_model = None
        
        # Pipeline state
        self.is_fitted = False
        self.feature_names = []
        self.pipeline_info = {}
        
        # Create preprocessor
        if preprocessing_config is None:
            self.preprocessor = create_optimal_preprocessor(task_type='image_reconstruction')
        else:
            self.preprocessor = EEGPreprocessor(**preprocessing_config)
        
        # Create feature extractor
        self.feature_extractor = UltraHighDimExtractor(
            target_dimensions=target_dimensions,
            **kwargs
        )
        
        logger.info(f"Initialized ImageReconstructionPipeline with {target_dimensions} target dimensions")
    
    def fit(self, eeg_data: np.ndarray, images: np.ndarray = None) -> 'ImageReconstructionPipeline':
        """
        Fit the complete pipeline.
        
        Args:
            eeg_data: Raw EEG data (n_samples, n_channels, n_timepoints)
            images: Target images (optional, for supervised training)
            
        Returns:
            self: Fitted pipeline
        """
        logger.info("Fitting ImageReconstructionPipeline...")
        
        # Validate input data
        eeg_data = validate_eeg_data(eeg_data)
        
        # Step 1: Fit preprocessor
        logger.info("Step 1: Fitting preprocessor...")
        start_time = time.time()
        self.preprocessor.fit(eeg_data)
        preprocess_time = time.time() - start_time
        logger.info(f"Preprocessor fitted in {preprocess_time:.2f}s")
        
        # Step 2: Apply preprocessing
        logger.info("Step 2: Applying preprocessing...")
        start_time = time.time()
        clean_eeg = self.preprocessor.transform(eeg_data)
        transform_time = time.time() - start_time
        logger.info(f"Preprocessing applied in {transform_time:.2f}s")
        
        # Step 3: Fit feature extractor
        logger.info("Step 3: Fitting feature extractor...")
        start_time = time.time()
        self.feature_extractor.fit(clean_eeg)
        fit_time = time.time() - start_time
        logger.info(f"Feature extractor fitted in {fit_time:.2f}s")
        
        # Step 4: Extract features
        logger.info("Step 4: Extracting features...")
        start_time = time.time()
        features = self.feature_extractor.transform(clean_eeg)
        extract_time = time.time() - start_time
        logger.info(f"Features extracted in {extract_time:.2f}s")
        logger.info(f"Extracted {features.shape[1]} features")
        
        # Step 5: Feature selection (optional)
        if self.feature_selection:
            logger.info("Step 5: Applying feature selection...")
            start_time = time.time()
            features, selected_indices = self._apply_feature_selection(features)
            selection_time = time.time() - start_time
            logger.info(f"Feature selection completed in {selection_time:.2f}s")
            logger.info(f"Selected {features.shape[1]} features")
        
        # Step 6: Train reconstruction model (if images provided)
        if images is not None:
            logger.info("Step 6: Training reconstruction model...")
            start_time = time.time()
            self._train_reconstruction_model(features, images)
            train_time = time.time() - start_time
            logger.info(f"Reconstruction model trained in {train_time:.2f}s")
        
        # Store pipeline info
        self.pipeline_info = {
            'n_samples': eeg_data.shape[0],
            'n_features_extracted': self.feature_extractor.n_features,
            'n_features_selected': features.shape[1] if self.feature_selection else self.feature_extractor.n_features,
            'preprocessing_time': preprocess_time,
            'extraction_time': extract_time,
            'total_time': preprocess_time + transform_time + fit_time + extract_time
        }
        
        self.is_fitted = True
        logger.info("ImageReconstructionPipeline fitted successfully")
        
        return self
    
    def extract_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract features from EEG data.
        
        Args:
            eeg_data: Raw EEG data
            
        Returns:
            np.ndarray: Extracted features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before feature extraction")
        
        # Validate input
        eeg_data = validate_eeg_data(eeg_data)
        
        # Apply preprocessing
        clean_eeg = self.preprocessor.transform(eeg_data)
        
        # Extract features
        features = self.feature_extractor.transform(clean_eeg)
        
        # Apply feature selection if fitted
        if self.feature_selection and self.feature_selector is not None:
            features = self.feature_selector.transform(features)
        
        return features
    
    def reconstruct_images(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Reconstruct images from EEG data.
        
        Args:
            eeg_data: Raw EEG data
            
        Returns:
            np.ndarray: Reconstructed images
        """
        if self.reconstruction_model is None:
            raise ValueError("Reconstruction model not trained")
        
        # Extract features
        features = self.extract_features(eeg_data)
        
        # Reconstruct images
        reconstructed = self.reconstruction_model.predict(features)
        
        return reconstructed
    
    def _apply_feature_selection(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply feature selection to reduce dimensionality."""
        from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
        from sklearn.pipeline import Pipeline
        
        # Create feature selection pipeline
        selectors = []
        
        # Remove zero variance features
        variance_selector = VarianceThreshold(threshold=0.01)
        selectors.append(('variance', variance_selector))
        
        # Select top features based on variance
        if features.shape[1] > 20000:
            k_best = SelectKBest(score_func=lambda X, y: np.var(X, axis=0), k=20000)
            selectors.append(('kbest', k_best))
        
        # Create pipeline
        self.feature_selector = Pipeline(selectors)
        
        # Fit and transform
        selected_features = self.feature_selector.fit_transform(features)
        
        # Get selected indices
        selected_indices = np.arange(features.shape[1])
        for name, selector in self.feature_selector.steps:
            if hasattr(selector, 'get_support'):
                selected_indices = selected_indices[selector.get_support()]
        
        return selected_features, selected_indices
    
    def _train_reconstruction_model(self, features: np.ndarray, images: np.ndarray):
        """Train image reconstruction model."""
        # Flatten images
        if images.ndim > 2:
            images_flat = images.reshape(images.shape[0], -1)
        else:
            images_flat = images
        
        # Create model based on type
        if self.model_type == 'mlp':
            from sklearn.neural_network import MLPRegressor
            self.reconstruction_model = MLPRegressor(
                hidden_layer_sizes=(1000, 500, 256),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        elif self.model_type == 'linear':
            from sklearn.linear_model import Ridge
            self.reconstruction_model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        self.reconstruction_model.fit(features, images_flat)
    
    def evaluate_reconstruction(self, eeg_data: np.ndarray, true_images: np.ndarray) -> Dict[str, float]:
        """
        Evaluate reconstruction quality.
        
        Args:
            eeg_data: Test EEG data
            true_images: True images
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Reconstruct images
        reconstructed = self.reconstruct_images(eeg_data)
        
        # Flatten true images if needed
        if true_images.ndim > 2:
            true_flat = true_images.reshape(true_images.shape[0], -1)
        else:
            true_flat = true_images
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score
        
        mse = mean_squared_error(true_flat, reconstructed)
        r2 = r2_score(true_flat, reconstructed)
        
        # Calculate correlation
        correlations = []
        for i in range(len(true_flat)):
            corr = np.corrcoef(true_flat[i], reconstructed[i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        mean_correlation = np.mean(correlations) if correlations else 0.0
        
        return {
            'mse': mse,
            'r2_score': r2,
            'mean_correlation': mean_correlation,
            'n_samples': len(true_flat)
        }
    
    def get_pipeline_info(self) -> Dict:
        """Get pipeline information and statistics."""
        info = self.pipeline_info.copy()
        
        if self.feature_extractor:
            info['extractor_info'] = self.feature_extractor.get_extractor_info()
            info['feature_breakdown'] = self.feature_extractor.get_feature_breakdown()
        
        if self.preprocessor:
            info['preprocessing_config'] = self.preprocessor.get_preprocessing_info()
        
        return info
    
    def save_pipeline(self, filepath: str):
        """Save fitted pipeline to file."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        pipeline_data = {
            'preprocessor': self.preprocessor,
            'feature_extractor': self.feature_extractor,
            'feature_selector': self.feature_selector,
            'reconstruction_model': self.reconstruction_model,
            'pipeline_info': self.pipeline_info,
            'config': {
                'target_dimensions': self.target_dimensions,
                'feature_selection': self.feature_selection,
                'model_type': self.model_type
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: str) -> 'ImageReconstructionPipeline':
        """Load fitted pipeline from file."""
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        # Create pipeline instance
        config = pipeline_data['config']
        pipeline = cls(**config)
        
        # Restore components
        pipeline.preprocessor = pipeline_data['preprocessor']
        pipeline.feature_extractor = pipeline_data['feature_extractor']
        pipeline.feature_selector = pipeline_data['feature_selector']
        pipeline.reconstruction_model = pipeline_data['reconstruction_model']
        pipeline.pipeline_info = pipeline_data['pipeline_info']
        pipeline.is_fitted = True
        
        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline
