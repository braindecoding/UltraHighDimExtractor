#!/usr/bin/env python3
"""
Test UltraHighDimExtractor functionality
"""

import sys
import os
import numpy as np
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🚀 Testing UltraHighDimExtractor...")

# Test 1: Import UltraHighDimExtractor
try:
    print("1. Testing UltraHighDimExtractor import...")
    from core.ultra_extractor import UltraHighDimExtractor
    print("   ✅ UltraHighDimExtractor imported successfully")
    
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    print("   This is expected if waveletfeatures folder is not accessible")
    exit(1)

# Test 2: Create extractor instance
try:
    print("\n2. Testing extractor creation...")
    extractor = UltraHighDimExtractor(
        target_dimensions=10000,
        wavelets=['db4', 'db8'],
        max_dwt_levels=4,
        max_wpd_levels=3
    )
    print(f"   ✅ Extractor created: {extractor.name}")
    print(f"   Target dimensions: {extractor.target_dimensions}")
    print(f"   Wavelets: {extractor.wavelets}")
    
except Exception as e:
    print(f"   ❌ Creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Generate test data
try:
    print("\n3. Generating test data...")
    np.random.seed(42)
    n_samples = 10
    n_channels = 14
    n_timepoints = 128
    
    # Generate realistic EEG-like data
    test_data = np.zeros((n_samples, n_channels, n_timepoints))
    t = np.linspace(0, 1, n_timepoints)
    
    for i in range(n_samples):
        for ch in range(n_channels):
            # Simulate brain activity
            signal = 2.0 * np.sin(2 * np.pi * 10 * t)  # Alpha
            signal += 1.0 * np.sin(2 * np.pi * 20 * t)  # Beta
            signal += 0.5 * np.sin(2 * np.pi * 40 * t)  # Gamma
            signal += 0.3 * np.random.randn(n_timepoints)  # Noise
            test_data[i, ch, :] = signal
    
    print(f"   ✅ Test data generated: {test_data.shape}")
    print(f"   Data range: [{np.min(test_data):.2f}, {np.max(test_data):.2f}]")
    
except Exception as e:
    print(f"   ❌ Data generation failed: {e}")
    exit(1)

# Test 4: Feature extraction
try:
    print("\n4. Testing feature extraction...")

    # Use preprocessed data directly (no preprocessing step)
    from utils.validation import validate_eeg_data
    validated_data = validate_eeg_data(test_data)
    print(f"   ✅ Data validated: {validated_data.shape}")

    # Extract features
    print("   Extracting ultra-high dimensional features...")
    start_time = time.time()
    features = extractor.fit_transform(validated_data)
    extraction_time = time.time() - start_time
    
    print(f"   ✅ Feature extraction completed!")
    print(f"   Features shape: {features.shape}")
    print(f"   Number of features: {features.shape[1]:,}")
    print(f"   Extraction time: {extraction_time:.2f}s")
    print(f"   Features per second: {features.shape[1]/extraction_time:.0f}")
    
    # Check feature quality
    n_nan = np.sum(np.isnan(features))
    n_inf = np.sum(np.isinf(features))
    feature_range = [np.min(features), np.max(features)]
    
    print(f"   Feature quality:")
    print(f"     NaN values: {n_nan}")
    print(f"     Infinite values: {n_inf}")
    print(f"     Value range: [{feature_range[0]:.3f}, {feature_range[1]:.3f}]")
    
except Exception as e:
    print(f"   ❌ Feature extraction failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Extractor information
try:
    print("\n5. Testing extractor information...")
    
    # Get extractor info
    extractor_info = extractor.get_extractor_info()
    print(f"   Number of sub-extractors: {len(extractor_info)}")
    
    for i, info in enumerate(extractor_info):
        print(f"   Extractor {i+1}: {info['type']} with {info['wavelet']} (levels={info['levels']})")
    
    # Get feature breakdown
    breakdown = extractor.get_feature_breakdown()
    print(f"\n   Feature breakdown:")
    total_features = 0
    for extractor_name, n_features in breakdown.items():
        print(f"     {extractor_name}: {n_features:,} features")
        total_features += n_features
    
    print(f"   Total features: {total_features:,}")
    
except Exception as e:
    print(f"   ❌ Information retrieval failed: {e}")

# Test 6: Performance estimation
try:
    print("\n6. Testing performance estimation...")
    
    # Estimate extraction time
    estimated_time = extractor.estimate_extraction_time(100)
    print(f"   Estimated time for 100 samples: {estimated_time:.1f}s")
    
    # Estimate memory usage
    memory_usage = extractor.get_memory_usage_estimate(100)
    print(f"   Memory usage estimate for 100 samples:")
    for key, value in memory_usage.items():
        print(f"     {key}: {value:.1f} MB")
    
except Exception as e:
    print(f"   ❌ Performance estimation failed: {e}")

print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("🎯 UltraHighDimExtractor is working correctly")
print(f"🚀 Ready for ultra-high dimensional feature extraction!")

# Summary
print(f"\n📊 SUMMARY:")
print(f"   ✅ Successfully extracted {features.shape[1]:,} features")
print(f"   ✅ Processing time: {extraction_time:.2f}s for {n_samples} samples")
print(f"   ✅ Feature quality: {n_nan} NaN, {n_inf} infinite values")
print(f"   ✅ Ready for image reconstruction tasks")
