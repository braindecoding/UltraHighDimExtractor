#!/usr/bin/env python3
"""
Test README Example
==================

Test the example code from README.md to ensure it works correctly.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_usage():
    """Test the basic usage example from README."""
    print("🧪 Testing README Basic Usage Example...")
    
    try:
        # Import as shown in README
        from core.ultra_extractor import UltraHighDimExtractor
        from utils.validation import validate_eeg_data
        
        # Generate sample preprocessed EEG data
        # Shape: (100, 14, 128) as mentioned in README
        eeg_data = np.random.randn(100, 14, 128)
        print(f"Input EEG shape: {eeg_data.shape}")
        
        # 1. Validate data format
        validated_data = validate_eeg_data(eeg_data)
        
        # 2. Extract ultra-high dimensional features
        extractor = UltraHighDimExtractor(target_dimensions=35000)
        features = extractor.fit_transform(validated_data)
        
        print(f"Output features shape: {features.shape}")
        print(f"Extracted {features.shape[1]:,} features per sample!")
        
        # Verify results match README expectations
        expected_samples = 100
        expected_min_features = 30000  # Should be around 35,672
        
        assert features.shape[0] == expected_samples, f"Expected {expected_samples} samples, got {features.shape[0]}"
        assert features.shape[1] >= expected_min_features, f"Expected at least {expected_min_features} features, got {features.shape[1]}"
        
        print("✅ README basic usage example works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ README example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_example():
    """Test the pipeline example from README."""
    print("\n🧪 Testing README Pipeline Example...")
    
    try:
        from core.pipeline import ImageReconstructionPipeline
        from utils.validation import validate_eeg_data
        
        # Generate sample preprocessed EEG data
        your_preprocessed_eeg_data = np.random.randn(50, 14, 128)
        
        # Complete pipeline for preprocessed data
        pipeline = ImageReconstructionPipeline(target_dimensions=35000)
        
        # Extract features from preprocessed EEG
        preprocessed_eeg = validate_eeg_data(your_preprocessed_eeg_data)
        features = pipeline.extract_features(preprocessed_eeg)
        
        print(f"Pipeline features shape: {features.shape}")
        print(f"Pipeline extracted {features.shape[1]:,} features per sample!")
        
        # Verify results
        assert features.shape[0] == 50, f"Expected 50 samples, got {features.shape[0]}"
        assert features.shape[1] >= 30000, f"Expected at least 30000 features, got {features.shape[1]}"
        
        print("✅ README pipeline example works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ README pipeline example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_configuration():
    """Test the advanced configuration example from README."""
    print("\n🧪 Testing README Advanced Configuration Example...")
    
    try:
        from core.ultra_extractor import UltraHighDimExtractor
        from utils.validation import validate_eeg_data
        
        # Custom configuration as shown in README
        extractor = UltraHighDimExtractor(
            target_dimensions=35000,
            wavelets=['db4', 'db8', 'coif3'],
            max_dwt_levels=6,
            max_wpd_levels=5,
            feature_types=['statistical', 'energy', 'entropy'],
            sampling_rate=128.0,
            optimize_for='image_reconstruction'
        )
        
        # Generate sample preprocessed data
        preprocessed_eeg_data = np.random.randn(20, 14, 128)
        validated_data = validate_eeg_data(preprocessed_eeg_data)
        
        # Extract features from preprocessed data
        features = extractor.fit_transform(validated_data)
        
        print(f"Advanced config features shape: {features.shape}")
        print(f"Advanced config extracted {features.shape[1]:,} features per sample!")
        
        # Verify results
        assert features.shape[0] == 20, f"Expected 20 samples, got {features.shape[0]}"
        assert features.shape[1] >= 25000, f"Expected at least 25000 features, got {features.shape[1]}"
        
        print("✅ README advanced configuration example works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ README advanced configuration example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing README Examples...")
    print("=" * 60)
    
    results = []
    
    # Test all examples
    results.append(test_basic_usage())
    results.append(test_pipeline_example())
    results.append(test_advanced_configuration())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL {total} README EXAMPLES PASSED!")
        print("🎉 README is accurate and up-to-date!")
    else:
        print(f"❌ {total - passed} out of {total} examples failed")
        print("⚠️ README needs updates")
    
    print("=" * 60)
