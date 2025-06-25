#!/usr/bin/env python3
"""
Simple import test for UltraHighDimWaveletExtractor package
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔧 Testing package imports...")

try:
    # Test basic import
    print("1. Testing basic package import...")
    import UltraHighDimWaveletExtractor
    print("   ✅ Package imported successfully")
    
    # Test version info
    print("2. Testing version info...")
    print(f"   Version: {UltraHighDimWaveletExtractor.__version__}")
    print(f"   Author: {UltraHighDimWaveletExtractor.__author__}")
    
    # Test info function
    print("3. Testing info function...")
    UltraHighDimWaveletExtractor.info()
    
    # Test core imports
    print("4. Testing core imports...")
    from UltraHighDimWaveletExtractor.core.base import WaveletFeatureBase
    print("   ✅ Base classes imported")
    
    from UltraHighDimWaveletExtractor.core.preprocessing import EEGPreprocessor
    print("   ✅ Preprocessor imported")
    
    # Test utils imports
    print("5. Testing utils imports...")
    from UltraHighDimWaveletExtractor.utils.validation import validate_eeg_data
    print("   ✅ Validation utils imported")
    
    from UltraHighDimWaveletExtractor.utils.metrics import FeatureQualityMetrics
    print("   ✅ Metrics utils imported")
    
    print("\n✅ ALL IMPORTS SUCCESSFUL!")
    print("Package is ready for use.")
    
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
