#!/usr/bin/env python3
"""
Simple test for package components
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔧 Testing individual components...")

# Test 1: Base classes
try:
    print("1. Testing base classes...")
    from core.base import WaveletFeatureBase, WaveletAnalyzer
    print("   ✅ Base classes imported successfully")
    
    # Test WaveletAnalyzer
    wavelets = WaveletAnalyzer.get_available_wavelets()
    print(f"   Available wavelet families: {len(wavelets)}")
    
    recommended = WaveletAnalyzer.recommend_wavelets_for_eeg()
    print(f"   Recommended for EEG: {recommended}")
    
except Exception as e:
    print(f"   ❌ Base classes failed: {e}")

# Test 2: Preprocessing
try:
    print("\n2. Testing preprocessing...")
    from core.preprocessing import EEGPreprocessor, create_optimal_preprocessor
    print("   ✅ Preprocessing imported successfully")
    
    # Test creating preprocessor
    preprocessor = create_optimal_preprocessor(task_type='image_reconstruction')
    print(f"   ✅ Created preprocessor: {preprocessor.__class__.__name__}")
    
except Exception as e:
    print(f"   ❌ Preprocessing failed: {e}")

# Test 3: Validation utilities
try:
    print("\n3. Testing validation utilities...")
    from utils.validation import validate_eeg_data, DataValidator
    print("   ✅ Validation utilities imported successfully")
    
    # Test validator
    validator = DataValidator()
    print(f"   ✅ Created validator: {validator.__class__.__name__}")
    
except Exception as e:
    print(f"   ❌ Validation utilities failed: {e}")

# Test 4: Metrics utilities
try:
    print("\n4. Testing metrics utilities...")
    from utils.metrics import FeatureQualityMetrics
    print("   ✅ Metrics utilities imported successfully")
    
except Exception as e:
    print(f"   ❌ Metrics utilities failed: {e}")

# Test 5: Simple functionality test
try:
    print("\n5. Testing basic functionality...")
    import numpy as np

    # Import functions we need
    from utils.validation import validate_eeg_data
    from utils.metrics import FeatureQualityMetrics

    # Generate test data
    test_data = np.random.randn(5, 14, 128)
    print(f"   Generated test data: {test_data.shape}")

    # Test validation
    validated_data = validate_eeg_data(test_data)
    print(f"   ✅ Data validation passed: {validated_data.shape}")

    # Test preprocessing
    preprocessor = create_optimal_preprocessor()
    clean_data = preprocessor.fit_transform(test_data)
    print(f"   ✅ Preprocessing passed: {clean_data.shape}")

    # Test quality metrics
    # Create dummy features for testing
    dummy_features = np.random.randn(5, 1000)
    metrics = FeatureQualityMetrics.compute_basic_metrics(dummy_features)
    print(f"   ✅ Quality metrics computed: {len(metrics)} metrics")

except Exception as e:
    print(f"   ❌ Functionality test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Component testing completed!")
print("🎯 Ready to implement main UltraHighDimExtractor")
