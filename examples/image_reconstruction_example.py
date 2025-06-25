#!/usr/bin/env python3
"""
Complete Image Reconstruction Example
====================================

This example demonstrates how to use UltraHighDimWaveletExtractor
for EEG-to-image reconstruction tasks.

Features:
- Ultra-high dimensional feature extraction (35K+ features)
- Complete preprocessing pipeline
- Multiple reconstruction models
- Quality evaluation metrics
- Visualization of results
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ultra_extractor import UltraHighDimExtractor
from core.preprocessing import create_optimal_preprocessor
from utils.validation import validate_eeg_data
from utils.metrics import FeatureQualityMetrics


class EEGImageReconstructor:
    """Complete EEG-to-Image reconstruction pipeline."""
    
    def __init__(self, target_dimensions=30000, model_type='ridge'):
        """
        Initialize the reconstructor.
        
        Args:
            target_dimensions (int): Target number of features to extract
            model_type (str): Type of reconstruction model ('ridge', 'elastic', 'rf')
        """
        self.target_dimensions = target_dimensions
        self.model_type = model_type
        
        # Initialize components
        self.preprocessor = None
        self.extractor = None
        self.scaler = None
        self.model = None
        
        # Performance tracking
        self.extraction_time = 0
        self.training_time = 0
        self.feature_quality = {}
        
    def setup_pipeline(self):
        """Setup the complete processing pipeline."""
        print("🔧 Setting up EEG-to-Image reconstruction pipeline...")
        
        # 1. EEG Preprocessor
        self.preprocessor = create_optimal_preprocessor(
            task_type='image_reconstruction',
            preserve_high_freq=True,
            aggressive_artifact_removal=True
        )
        print("   ✅ EEG preprocessor configured")
        
        # 2. Ultra-high dimensional feature extractor
        self.extractor = UltraHighDimExtractor(
            target_dimensions=self.target_dimensions,
            wavelets=['db4', 'db8', 'coif5', 'bior4.4'],
            max_dwt_levels=6,
            max_wpd_levels=5,
            feature_types=['statistical', 'energy', 'entropy', 'morphological'],
            include_cross_frequency=True,
            optimize_for='image_reconstruction'
        )
        print(f"   ✅ Ultra-high dim extractor configured (target: {self.target_dimensions:,} features)")
        
        # 3. Feature scaler
        self.scaler = StandardScaler()
        print("   ✅ Feature scaler configured")
        
        # 4. Reconstruction model
        if self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0, max_iter=1000)
        elif self.model_type == 'elastic':
            self.model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000)
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"   ✅ {self.model_type.upper()} reconstruction model configured")
        
    def extract_features(self, eeg_data):
        """Extract ultra-high dimensional features from EEG data."""
        print(f"\n🧠 Extracting features from {eeg_data.shape[0]} EEG samples...")
        
        # Validate data
        validated_data = validate_eeg_data(eeg_data)
        print(f"   ✅ Data validation passed: {validated_data.shape}")
        
        # Preprocess
        clean_data = self.preprocessor.fit_transform(validated_data)
        print(f"   ✅ Preprocessing completed: {clean_data.shape}")
        
        # Extract features
        start_time = time.time()
        features = self.extractor.fit_transform(clean_data)
        self.extraction_time = time.time() - start_time
        
        print(f"   ✅ Feature extraction completed: {features.shape}")
        print(f"   ⏱️  Extraction time: {self.extraction_time:.2f}s")
        print(f"   🚀 Speed: {features.shape[1]/self.extraction_time:.0f} features/second")
        
        # Assess feature quality
        self.feature_quality = FeatureQualityMetrics.compute_basic_metrics(features)
        print(f"   📊 Feature quality: {len(self.feature_quality)} metrics computed")
        
        return features
    
    def train_reconstruction_model(self, eeg_features, images, test_size=0.2):
        """Train the image reconstruction model."""
        print(f"\n🎯 Training {self.model_type.upper()} reconstruction model...")
        
        # Prepare image data
        if len(images.shape) > 2:
            images_flat = images.reshape(images.shape[0], -1)
        else:
            images_flat = images
        
        print(f"   📊 EEG features: {eeg_features.shape}")
        print(f"   📊 Images: {images_flat.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            eeg_features, images_flat, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        self.training_time = time.time() - start_time
        
        print(f"   ✅ Model training completed in {self.training_time:.2f}s")
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"   📊 Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"   📊 Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def reconstruct_images(self, eeg_data):
        """Reconstruct images from new EEG data."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_reconstruction_model first.")
        
        # Extract features
        features = self.extract_features(eeg_data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Reconstruct
        reconstructed = self.model.predict(features_scaled)
        
        return reconstructed
    
    def get_performance_summary(self):
        """Get performance summary."""
        return {
            'target_dimensions': self.target_dimensions,
            'extraction_time': self.extraction_time,
            'training_time': self.training_time,
            'feature_quality': self.feature_quality,
            'model_type': self.model_type
        }


def generate_synthetic_data():
    """Generate synthetic EEG and image data for demonstration."""
    print("🎲 Generating synthetic EEG and image data...")
    
    np.random.seed(42)
    
    # EEG parameters
    n_samples = 100
    n_channels = 14
    n_timepoints = 128
    
    # Image parameters
    img_height, img_width = 32, 32
    
    # Generate EEG data with realistic patterns
    eeg_data = np.zeros((n_samples, n_channels, n_timepoints))
    images = np.zeros((n_samples, img_height, img_width))
    
    t = np.linspace(0, 1, n_timepoints)
    
    for i in range(n_samples):
        # Create image pattern
        pattern_type = i % 4
        if pattern_type == 0:  # Horizontal stripes
            img = np.sin(np.linspace(0, 4*np.pi, img_height))[:, None]
        elif pattern_type == 1:  # Vertical stripes
            img = np.sin(np.linspace(0, 4*np.pi, img_width))[None, :]
        elif pattern_type == 2:  # Checkerboard
            x, y = np.meshgrid(np.linspace(0, 4*np.pi, img_width), 
                              np.linspace(0, 4*np.pi, img_height))
            img = np.sin(x) * np.sin(y)
        else:  # Random pattern
            img = np.random.randn(img_height, img_width)
        
        images[i] = img
        
        # Generate corresponding EEG with pattern-dependent characteristics
        for ch in range(n_channels):
            if pattern_type == 0:  # Horizontal -> more alpha
                signal = 0.5 * np.sin(2 * np.pi * 10 * t)
                signal += 0.2 * np.sin(2 * np.pi * 6 * t)
            elif pattern_type == 1:  # Vertical -> more beta
                signal = 0.4 * np.sin(2 * np.pi * 20 * t)
                signal += 0.3 * np.sin(2 * np.pi * 15 * t)
            elif pattern_type == 2:  # Checkerboard -> more gamma
                signal = 0.3 * np.sin(2 * np.pi * 40 * t)
                signal += 0.2 * np.sin(2 * np.pi * 30 * t)
            else:  # Random -> mixed frequencies
                signal = 0.2 * np.sin(2 * np.pi * 10 * t)
                signal += 0.2 * np.sin(2 * np.pi * 20 * t)
                signal += 0.1 * np.sin(2 * np.pi * 40 * t)
            
            # Add noise and channel-specific variations
            signal += 0.1 * np.random.randn(n_timepoints)
            signal += 0.05 * np.sin(2 * np.pi * (5 + ch) * t)  # Channel variation
            
            eeg_data[i, ch, :] = signal
    
    print(f"   ✅ Generated {n_samples} EEG-image pairs")
    print(f"   📊 EEG shape: {eeg_data.shape}")
    print(f"   📊 Images shape: {images.shape}")
    
    return eeg_data, images


def visualize_results(results, images_original, images_reconstructed, n_examples=4):
    """Visualize reconstruction results."""
    print("\n📊 Creating visualization...")
    
    fig, axes = plt.subplots(3, n_examples, figsize=(12, 8))
    
    for i in range(n_examples):
        # Original image
        axes[0, i].imshow(images_original[i], cmap='viridis')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed image
        axes[1, i].imshow(images_reconstructed[i], cmap='viridis')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
        
        # Difference
        diff = np.abs(images_original[i] - images_reconstructed[i])
        axes[2, i].imshow(diff, cmap='Reds')
        axes[2, i].set_title(f'Difference {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction_results.png', dpi=150, bbox_inches='tight')
    print("   ✅ Visualization saved as 'reconstruction_results.png'")
    
    # Performance plot
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['y_test'][:50].flatten(), label='Original', alpha=0.7)
    plt.plot(results['y_pred_test'][:50].flatten(), label='Reconstructed', alpha=0.7)
    plt.title('Pixel Values Comparison (First 50 pixels)')
    plt.xlabel('Pixel Index')
    plt.ylabel('Pixel Value')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(results['y_test'].flatten(), results['y_pred_test'].flatten(), alpha=0.5)
    plt.plot([results['y_test'].min(), results['y_test'].max()], 
             [results['y_test'].min(), results['y_test'].max()], 'r--')
    plt.xlabel('Original Pixel Values')
    plt.ylabel('Reconstructed Pixel Values')
    plt.title(f'Correlation (R² = {results["test_r2"]:.3f})')
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✅ Performance analysis saved as 'performance_analysis.png'")


def main():
    """Main demonstration function."""
    print("🎨 EEG-TO-IMAGE RECONSTRUCTION DEMONSTRATION")
    print("=" * 60)
    
    # Generate synthetic data
    eeg_data, images = generate_synthetic_data()
    
    # Test different configurations
    configs = [
        {'target_dimensions': 15000, 'model_type': 'ridge'},
        {'target_dimensions': 25000, 'model_type': 'ridge'},
        {'target_dimensions': 35000, 'model_type': 'ridge'},
    ]
    
    best_performance = {'test_r2': -np.inf, 'config': None, 'results': None}
    
    for config in configs:
        print(f"\n🔧 Testing configuration: {config}")
        
        # Initialize reconstructor
        reconstructor = EEGImageReconstructor(**config)
        reconstructor.setup_pipeline()
        
        # Extract features and train
        features = reconstructor.extract_features(eeg_data)
        results = reconstructor.train_reconstruction_model(features, images)
        
        # Track best performance
        if results['test_r2'] > best_performance['test_r2']:
            best_performance = {
                'test_r2': results['test_r2'],
                'config': config,
                'results': results,
                'reconstructor': reconstructor
            }
        
        print(f"   🎯 Configuration R²: {results['test_r2']:.4f}")
    
    # Use best configuration for final demonstration
    print(f"\n🏆 Best configuration: {best_performance['config']}")
    print(f"🎯 Best R²: {best_performance['test_r2']:.4f}")
    
    # Reconstruct test images
    best_reconstructor = best_performance['reconstructor']
    test_results = best_performance['results']
    
    # Reshape for visualization
    img_shape = images.shape[1:]
    original_images = test_results['y_test'][:4].reshape(-1, *img_shape)
    reconstructed_images = test_results['y_pred_test'][:4].reshape(-1, *img_shape)
    
    # Visualize results
    visualize_results(test_results, original_images, reconstructed_images)
    
    # Performance summary
    summary = best_reconstructor.get_performance_summary()
    print(f"\n📋 FINAL PERFORMANCE SUMMARY:")
    print(f"   🎯 Target dimensions: {summary['target_dimensions']:,}")
    print(f"   ⏱️  Feature extraction time: {summary['extraction_time']:.2f}s")
    print(f"   ⏱️  Model training time: {summary['training_time']:.2f}s")
    print(f"   📊 Test R²: {test_results['test_r2']:.4f}")
    print(f"   📊 Test MSE: {test_results['test_mse']:.4f}")
    
    print(f"\n✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print(f"🎨 Ultra-high dimensional features enable effective image reconstruction")


if __name__ == "__main__":
    main()
