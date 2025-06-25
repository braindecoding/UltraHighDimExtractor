# Data Format Specification

## Overview

The UltraHighDimWaveletExtractor package is designed to work with **preprocessed EEG data**. This document specifies the exact format and requirements for input data.

## ⚠️ Important Note

**This package does NOT include preprocessing functionality.** You must preprocess your EEG data using a dedicated preprocessing package before using this extractor.

## Required Data Format

### Input Shape
```
(n_samples, n_channels, n_timepoints)
```

- **n_samples**: Number of EEG trials/epochs
- **n_channels**: Number of EEG electrodes/channels
- **n_timepoints**: Number of time samples per trial

### Data Type
- **Type**: `numpy.ndarray`
- **Dtype**: `float32` or `float64`
- **Range**: Normalized values (recommended: [-1, 1] or [0, 1])

### Example Shapes
```python
# Common configurations
eeg_data.shape = (100, 14, 128)   # 100 trials, 14 channels, 128 timepoints
eeg_data.shape = (500, 32, 256)   # 500 trials, 32 channels, 256 timepoints
eeg_data.shape = (1000, 64, 512)  # 1000 trials, 64 channels, 512 timepoints
```

## Preprocessing Requirements

Before using this package, ensure your EEG data has been:

### 1. Artifact Removal
- ✅ Eye blinks removed (ICA, regression, or manual)
- ✅ Muscle artifacts removed
- ✅ Bad channels interpolated or removed
- ✅ Bad epochs rejected

### 2. Filtering
- ✅ High-pass filtered (typically 0.1-1 Hz)
- ✅ Low-pass filtered (typically 40-100 Hz)
- ✅ Notch filtered (50/60 Hz power line noise)

### 3. Normalization/Scaling
- ✅ Baseline corrected
- ✅ Scaled/normalized (z-score, min-max, or robust scaling)
- ✅ Consistent scaling across channels and trials

### 4. Epoching
- ✅ Data segmented into trials/epochs
- ✅ Consistent epoch length
- ✅ Proper time alignment

## Data Quality Checks

The package includes validation that checks for:

```python
from utils.validation import validate_eeg_data

# This will check:
# - Correct shape (3D array)
# - No NaN or infinite values
# - Reasonable value ranges
# - Consistent dimensions
validated_data = validate_eeg_data(eeg_data)
```

## Example: Loading Preprocessed Data

```python
import numpy as np
from utils.validation import validate_eeg_data
from core.ultra_extractor import UltraHighDimExtractor

# Load your preprocessed data
eeg_data = np.load('preprocessed_eeg_data.npy')
print(f"Data shape: {eeg_data.shape}")
print(f"Data range: [{eeg_data.min():.3f}, {eeg_data.max():.3f}]")

# Validate format
validated_data = validate_eeg_data(eeg_data)

# Extract features
extractor = UltraHighDimExtractor(target_dimensions=30000)
features = extractor.fit_transform(validated_data)
```

## Recommended Preprocessing Packages

For preprocessing your raw EEG data, consider these packages:

### Python
- **MNE-Python**: Comprehensive EEG/MEG analysis
- **EEGLAB-Python**: Python wrapper for EEGLAB
- **Braindecode**: Deep learning for EEG
- **PyEEG**: Basic EEG processing

### MATLAB
- **EEGLAB**: Popular EEG analysis toolbox
- **FieldTrip**: Advanced neurophysiological data analysis
- **Brainstorm**: MEG/EEG analysis application

### R
- **eegkit**: EEG analysis toolkit
- **EEGUtils**: EEG data processing utilities

## Common Issues and Solutions

### Issue: "Data contains NaN values"
**Solution**: Check your preprocessing pipeline for:
- Incomplete artifact removal
- Bad channel interpolation failures
- Filtering edge effects

### Issue: "Data range seems unusual"
**Solution**: Ensure proper normalization:
```python
# Example normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
```

### Issue: "Inconsistent epoch lengths"
**Solution**: Ensure all epochs have the same number of timepoints:
```python
# Check epoch lengths
epoch_lengths = [epoch.shape[-1] for epoch in epochs]
assert len(set(epoch_lengths)) == 1, "Inconsistent epoch lengths"
```

## Performance Considerations

### Optimal Data Characteristics
- **Sampling Rate**: 128-512 Hz (higher rates increase computation time)
- **Epoch Length**: 0.5-2 seconds (balance between information and computation)
- **Number of Channels**: 14-64 channels (more channels = more features)
- **Data Type**: `float32` (reduces memory usage vs `float64`)

### Memory Usage
```python
# Estimate memory usage
n_samples, n_channels, n_timepoints = eeg_data.shape
memory_mb = (n_samples * n_channels * n_timepoints * 4) / (1024**2)  # float32
print(f"Estimated memory usage: {memory_mb:.1f} MB")
```

## Validation Checklist

Before using the extractor, verify:

- [ ] Data is 3D numpy array
- [ ] Shape is (n_samples, n_channels, n_timepoints)
- [ ] No NaN or infinite values
- [ ] Data is properly normalized
- [ ] Artifacts have been removed
- [ ] Appropriate filtering has been applied
- [ ] Epochs are consistently sized
- [ ] Data type is float32 or float64

## Contact and Support

If you have questions about data format or preprocessing:
1. Check the validation error messages
2. Review this specification
3. Consult preprocessing package documentation
4. Open an issue on the project repository
