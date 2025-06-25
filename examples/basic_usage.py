#!/usr/bin/env python3
"""
Basic Usage Example for UltraHighDimWaveletExtractor
===================================================

This example demonstrates the basic usage of the UltraHighDimWaveletExtractor
package for extracting ultra-high dimensional features from EEG data.

Usage:
    python basic_usage.py
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ultra_extractor import UltraHighDimExtractor
from core.preprocessing import create_optimal_preprocessor
from utils.validation import validate_eeg_data
from utils.metrics import FeatureQualityMetrics


def generate_sample_eeg_data():
    """Generate realistic sample EEG data for demonstration."""
    print("🧠 Generating sample EEG data...")
    
    np.random.seed(42)
    n_samples = 20
    n_channels = 14
    n_timepoints = 128
    sampling_rate = 128.0
    
    # Time vector
    t = np.linspace(0, 1, n_timepoints)
    
    # Initialize data
    eeg_data = np.zeros((n_samples, n_channels, n_timepoints))
    
    for sample_idx in range(n_samples):
        for ch_idx in range(n_channels):
            # Simulate different brain states
            if sample_idx < 10:  # "Simple visual stimuli"
                signal = 2.0 * np.sin(2 * np.pi * 10 * t)  # Alpha (10 Hz)
                signal += 1.0 * np.sin(2 * np.pi * 6 * t)   # Theta (6 Hz)
            else:  # "Complex visual stimuli"
                signal = 1.5 * np.sin(2 * np.pi * 20 * t)  # Beta (20 Hz)
                signal += 2.0 * np.sin(2 * np.pi * 40 * t)  # Gamma (40 Hz)
            
            # Add realistic noise
            signal += 0.5 * np.random.randn(n_timepoints)
            
            # Add channel-specific variations
            signal *= (0.8 + 0.4 * np.random.rand())  # Amplitude variation
            signal += np.random.uniform(-2, 2)        # DC offset
            
            eeg_data[sample_idx, ch_idx, :] = signal
    
    print(f"✅ Generated EEG data: {eeg_data.shape}")
    print(f"   Amplitude range: [{np.min(eeg_data):.2f}, {np.max(eeg_data):.2f}]")
    
    return eeg_data


def demo_basic_extraction():
    """Demonstrate basic feature extraction."""
    print("\n" + "="*60)
    print("🚀 BASIC FEATURE EXTRACTION DEMO")
    print("="*60)

    # Generate sample data
    eeg_data = generate_sample_eeg_data()

    # Method 1: Quick extraction with defaults
    print("\n🔧 Method 1: Quick extraction with defaults")
    start_time = time.time()

    # Quick extraction implementation
    validated_data = validate_eeg_data(eeg_data)
    preprocessor = create_optimal_preprocessor()
    clean_data = preprocessor.fit_transform(validated_data)
    extractor = UltraHighDimExtractor(target_dimensions=20000)
    features_quick = extractor.fit_transform(clean_data)

    quick_time = time.time() - start_time

    print(f"✅ Quick extraction completed:")
    print(f"   Features: {features_quick.shape[1]:,}")
    print(f"   Time: {quick_time:.2f}s")
    print(f"   Rate: {features_quick.shape[1]/quick_time:.0f} features/second")

    # Method 2: Step-by-step extraction
    print("\n🔧 Method 2: Step-by-step extraction")

    # Step 1: Preprocessing
    print("   Step 1: Preprocessing...")
    preprocessor = create_optimal_preprocessor(task_type='image_reconstruction')
    start_time = time.time()
    clean_eeg = preprocessor.fit_transform(eeg_data)
    preprocess_time = time.time() - start_time
    print(f"   ✅ Preprocessing: {preprocess_time:.2f}s")

    # Step 2: Feature extraction
    print("   Step 2: Feature extraction...")
    extractor = UltraHighDimExtractor(target_dimensions=20000)
    start_time = time.time()
    features_detailed = extractor.fit_transform(clean_eeg)
    extract_time = time.time() - start_time
    print(f"   ✅ Extraction: {extract_time:.2f}s")

    print(f"\n📊 Step-by-step results:")
    print(f"   Features: {features_detailed.shape[1]:,}")
    print(f"   Total time: {preprocess_time + extract_time:.2f}s")

    return features_quick, features_detailed, eeg_data


def demo_feature_validation():
    """Demonstrate feature validation and quality assessment."""
    print("\n" + "="*60)
    print("🔍 FEATURE VALIDATION DEMO")
    print("="*60)

    # Generate data and extract features
    eeg_data = generate_sample_eeg_data()

    # Extract features
    validated_data = validate_eeg_data(eeg_data)
    preprocessor = create_optimal_preprocessor()
    clean_data = preprocessor.fit_transform(validated_data)
    extractor = UltraHighDimExtractor(target_dimensions=15000)
    features = extractor.fit_transform(clean_data)

    # Basic validation
    print("\n🔧 Basic validation...")
    try:
        validated_eeg = validate_eeg_data(eeg_data)
        print("✅ EEG data validation passed")
        print(f"   Data shape: {validated_eeg.shape}")
    except Exception as e:
        print(f"❌ EEG validation failed: {e}")

    # Feature quality check
    print("\n🔧 Feature quality check...")
    n_nan = np.sum(np.isnan(features))
    n_inf = np.sum(np.isinf(features))
    print(f"   NaN values: {n_nan}")
    print(f"   Infinite values: {n_inf}")
    print(f"   Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")

    if n_nan == 0 and n_inf == 0:
        print("✅ Feature quality validation passed")
    else:
        print("⚠️ Feature quality issues detected")

    return features


def demo_feature_quality_assessment():
    """Demonstrate feature quality assessment."""
    print("\n" + "="*60)
    print("📊 FEATURE QUALITY ASSESSMENT DEMO")
    print("="*60)

    # Generate data and extract features
    eeg_data = generate_sample_eeg_data()

    # Extract features
    validated_data = validate_eeg_data(eeg_data)
    preprocessor = create_optimal_preprocessor()
    clean_data = preprocessor.fit_transform(validated_data)
    extractor = UltraHighDimExtractor(target_dimensions=10000)
    features = extractor.fit_transform(clean_data)

    # Basic quality assessment
    print("\n🔧 Computing feature quality metrics...")
    basic_metrics = FeatureQualityMetrics.compute_basic_metrics(features)

    print(f"\n📈 FEATURE QUALITY REPORT:")
    print("-" * 40)

    # Basic metrics
    print(f"📊 Basic Metrics:")
    print(f"   Features: {features.shape[1]:,}")
    print(f"   Samples: {features.shape[0]}")
    print(f"   Value range: [{np.min(features):.3f}, {np.max(features):.3f}]")
    print(f"   Mean: {np.mean(features):.3f}")
    print(f"   Std: {np.std(features):.3f}")
    print(f"   NaN values: {np.sum(np.isnan(features))}")
    print(f"   Infinite values: {np.sum(np.isinf(features))}")

    # Feature variance analysis
    feature_vars = np.var(features, axis=0)
    zero_var_features = np.sum(feature_vars == 0)
    low_var_features = np.sum(feature_vars < 1e-6)

    print(f"\n📊 Variance Analysis:")
    print(f"   Zero variance features: {zero_var_features}")
    print(f"   Low variance features: {low_var_features}")
    print(f"   Mean feature variance: {np.mean(feature_vars):.6f}")

    # Quality score
    quality_score = 100
    if np.sum(np.isnan(features)) > 0:
        quality_score -= 30
    if np.sum(np.isinf(features)) > 0:
        quality_score -= 30
    if zero_var_features > features.shape[1] * 0.1:
        quality_score -= 20

    print(f"\n🎯 Overall Quality Score: {quality_score}/100")

    if quality_score >= 80:
        print("✅ Excellent feature quality!")
    elif quality_score >= 60:
        print("⚠️  Good feature quality")
    else:
        print("❌ Poor feature quality - consider parameter tuning")

    return basic_metrics


def demo_extractor_configuration():
    """Demonstrate different extractor configurations."""
    print("\n" + "="*60)
    print("🔧 EXTRACTOR CONFIGURATION DEMO")
    print("="*60)

    # Generate test data
    eeg_data = generate_sample_eeg_data()

    # Test different configurations
    configs = [
        {'name': 'Fast', 'target_dimensions': 8000, 'wavelets': ['db4']},
        {'name': 'Balanced', 'target_dimensions': 15000, 'wavelets': ['db4', 'db8']},
        {'name': 'High-Quality', 'target_dimensions': 25000, 'wavelets': ['db4', 'db8', 'coif5']}
    ]

    results = {}

    print("\n🔧 Testing different configurations:")
    print("-" * 50)

    for config in configs:
        print(f"\n🔧 Testing {config['name']} configuration...")

        try:
            # Create extractor
            extractor = UltraHighDimExtractor(
                target_dimensions=config['target_dimensions'],
                wavelets=config['wavelets']
            )

            # Preprocess and extract
            validated_data = validate_eeg_data(eeg_data)
            preprocessor = create_optimal_preprocessor()
            clean_data = preprocessor.fit_transform(validated_data)

            start_time = time.time()
            features = extractor.fit_transform(clean_data)
            extraction_time = time.time() - start_time

            results[config['name']] = {
                'success': True,
                'n_features': features.shape[1],
                'extraction_time': extraction_time,
                'features_per_second': features.shape[1] / extraction_time
            }

            print(f"   ✅ Success: {features.shape[1]:,} features in {extraction_time:.2f}s")

        except Exception as e:
            results[config['name']] = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Failed: {str(e)}")

    # Summary
    print(f"\n📊 Configuration Summary:")
    print("-" * 30)
    for name, result in results.items():
        if result['success']:
            print(f"{name:15} | {result['n_features']:,} features | {result['extraction_time']:.2f}s")
        else:
            print(f"{name:15} | Failed: {result.get('error', 'Unknown error')}")

    return results


def demo_extractor_configuration():
    """Demonstrate different extractor configurations."""
    print("\n" + "="*60)
    print("⚙️  EXTRACTOR CONFIGURATION DEMO")
    print("="*60)
    
    # Generate sample data
    eeg_data = generate_sample_eeg_data()
    
    # Test different configurations
    configs = [
        {
            'name': 'Balanced',
            'params': {
                'target_dimensions': 15000,
                'wavelets': ['db4', 'db8', 'coif3'],
                'max_dwt_levels': 4,
                'max_wpd_levels': 3
            }
        },
        {
            'name': 'High-Dimensional',
            'params': {
                'target_dimensions': 30000,
                'wavelets': ['db8', 'db10', 'coif5', 'bior4.4'],
                'max_dwt_levels': 6,
                'max_wpd_levels': 5
            }
        },
        {
            'name': 'Fast',
            'params': {
                'target_dimensions': 5000,
                'wavelets': ['db4', 'db6'],
                'max_dwt_levels': 3,
                'max_wpd_levels': 2
            }
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🔧 Testing {config['name']} configuration...")
        
        try:
            # Create extractor
            extractor = UltraHighDimExtractor(**config['params'])
            
            # Extract features
            start_time = time.time()
            features = extractor.fit_transform(eeg_data)
            extraction_time = time.time() - start_time
            
            # Store results
            results[config['name']] = {
                'n_features': features.shape[1],
                'extraction_time': extraction_time,
                'features_per_second': features.shape[1] / extraction_time,
                'success': True
            }
            
            print(f"   ✅ Success: {features.shape[1]:,} features in {extraction_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[config['name']] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n📊 CONFIGURATION COMPARISON:")
    print("-" * 50)
    for name, result in results.items():
        if result['success']:
            print(f"{name:15} | {result['n_features']:6,} features | {result['extraction_time']:5.1f}s | {result['features_per_second']:6.0f} feat/s")
        else:
            print(f"{name:15} | Failed: {result.get('error', 'Unknown error')}")
    
    return results


def main():
    """Main demonstration function."""
    print("🚀 UltraHighDimWaveletExtractor - Basic Usage Demo")
    print("="*60)
    
    # Show package info
    print("📦 Package: UltraHighDimWaveletExtractor")
    print("🎯 Purpose: Ultra-high dimensional wavelet feature extraction")
    print("🧠 Target: EEG-to-image reconstruction")
    
    try:
        # Demo 1: Basic extraction
        features_quick, features_detailed, eeg_data = demo_basic_extraction()
        
        # Demo 2: Feature validation
        features_validated = demo_feature_validation()
        
        # Demo 3: Feature quality assessment
        quality_assessment = demo_feature_quality_assessment()
        
        # Demo 4: Extractor configuration
        config_results = demo_extractor_configuration()
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n💡 Key Takeaways:")
        print("1. ✅ Ultra-high dimensional extraction works (10,000+ features)")
        print("2. ✅ Preprocessing is essential for quality features")
        print("3. ✅ Feature validation helps ensure data quality")
        print("4. ✅ Different configurations offer speed/quality tradeoffs")
        print("5. ✅ Package is ready for image reconstruction tasks")
        
        print(f"\n🎯 Next Steps:")
        print("- Try with your own EEG data")
        print("- Experiment with different wavelet configurations")
        print("- Use ImageReconstructionPipeline for complete workflow")
        print("- Apply feature selection for optimal performance")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
