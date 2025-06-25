# UltraHighDimWaveletExtractor API Reference

## 🎯 Core Classes

### UltraHighDimExtractor

The main class for ultra-high dimensional wavelet feature extraction.

```python
class UltraHighDimExtractor(WaveletFeatureBase)
```

#### Constructor

```python
UltraHighDimExtractor(
    target_dimensions=25000,
    wavelets=['db8', 'db10', 'coif5', 'bior4.4', 'sym8'],
    max_dwt_levels=6,
    max_wpd_levels=5,
    feature_types=['statistical', 'energy', 'entropy', 'morphological'],
    sampling_rate=128,
    optimize_for='image_reconstruction',
    n_jobs=1,
    random_state=42
)
```

**Parameters:**
- `target_dimensions` (int): Target number of features to extract (default: 25000)
- `wavelets` (list): List of wavelet names to use (default: ['db8', 'db10', 'coif5', 'bior4.4', 'sym8'])
- `max_dwt_levels` (int): Maximum DWT decomposition levels (default: 6)
- `max_wpd_levels` (int): Maximum WPD decomposition levels (default: 5)
- `feature_types` (list): Types of features to extract (default: ['statistical', 'energy', 'entropy', 'morphological'])
- `sampling_rate` (float): EEG sampling rate in Hz (default: 128)
- `optimize_for` (str): Optimization target ('image_reconstruction', 'classification', 'general')
- `n_jobs` (int): Number of parallel jobs (default: 1, -1 for all cores)
- `random_state` (int): Random seed for reproducibility

#### Methods

##### fit_transform(X)
Extract ultra-high dimensional features from EEG data.

```python
features = extractor.fit_transform(X)
```

**Parameters:**
- `X` (ndarray): EEG data with shape (n_samples, n_channels, n_timepoints)

**Returns:**
- `features` (ndarray): Extracted features with shape (n_samples, n_features)

##### get_extractor_info()
Get information about sub-extractors.

```python
info = extractor.get_extractor_info()
```

**Returns:**
- `info` (list): List of dictionaries containing extractor information

##### get_feature_breakdown()
Get breakdown of features by extractor type.

```python
breakdown = extractor.get_feature_breakdown()
```

**Returns:**
- `breakdown` (dict): Dictionary mapping extractor names to feature counts

##### estimate_extraction_time(n_samples)
Estimate feature extraction time.

```python
time_est = extractor.estimate_extraction_time(100)
```

**Parameters:**
- `n_samples` (int): Number of samples to estimate for

**Returns:**
- `time_est` (float): Estimated time in seconds

##### get_memory_usage_estimate(n_samples)
Estimate memory usage for feature extraction.

```python
memory_est = extractor.get_memory_usage_estimate(100)
```

**Parameters:**
- `n_samples` (int): Number of samples to estimate for

**Returns:**
- `memory_est` (dict): Dictionary with memory estimates in MB

### EEGPreprocessor

Advanced EEG preprocessing optimized for image reconstruction.

```python
class EEGPreprocessor
```

#### Constructor

```python
EEGPreprocessor(
    lowpass_freq=59.0,
    highpass_freq=0.1,
    notch_freq=50.0,
    sampling_rate=128,
    remove_artifacts=True,
    normalize=True,
    preserve_high_freq=False
)
```

**Parameters:**
- `lowpass_freq` (float): Low-pass filter frequency in Hz
- `highpass_freq` (float): High-pass filter frequency in Hz
- `notch_freq` (float): Notch filter frequency in Hz (power line)
- `sampling_rate` (float): EEG sampling rate in Hz
- `remove_artifacts` (bool): Whether to remove artifacts
- `normalize` (bool): Whether to normalize data
- `preserve_high_freq` (bool): Whether to preserve high frequencies for image reconstruction

#### Methods

##### fit_transform(X)
Preprocess EEG data.

```python
clean_data = preprocessor.fit_transform(X)
```

**Parameters:**
- `X` (ndarray): Raw EEG data with shape (n_samples, n_channels, n_timepoints)

**Returns:**
- `clean_data` (ndarray): Preprocessed EEG data with same shape

## 🛠️ Utility Functions

### create_optimal_preprocessor(task_type)

Create an optimally configured preprocessor for specific tasks.

```python
from core.preprocessing import create_optimal_preprocessor

preprocessor = create_optimal_preprocessor(task_type='image_reconstruction')
```

**Parameters:**
- `task_type` (str): Type of task ('image_reconstruction', 'classification', 'general')

**Returns:**
- `preprocessor` (EEGPreprocessor): Configured preprocessor instance

### validate_eeg_data(data)

Validate EEG data format and content.

```python
from utils.validation import validate_eeg_data

validated_data = validate_eeg_data(data)
```

**Parameters:**
- `data` (ndarray): EEG data to validate

**Returns:**
- `validated_data` (ndarray): Validated EEG data

**Raises:**
- `ValueError`: If data format is invalid

### FeatureQualityMetrics

Static class for computing feature quality metrics.

#### compute_basic_metrics(features)

Compute basic quality metrics for extracted features.

```python
from utils.metrics import FeatureQualityMetrics

metrics = FeatureQualityMetrics.compute_basic_metrics(features)
```

**Parameters:**
- `features` (ndarray): Extracted features with shape (n_samples, n_features)

**Returns:**
- `metrics` (dict): Dictionary containing quality metrics

#### compute_comprehensive_metrics(features)

Compute comprehensive quality metrics including advanced statistics.

```python
metrics = FeatureQualityMetrics.compute_comprehensive_metrics(features)
```

**Parameters:**
- `features` (ndarray): Extracted features

**Returns:**
- `metrics` (dict): Dictionary with comprehensive metrics including:
  - `snr`: Signal-to-noise ratio estimate
  - `stability`: Feature stability across samples
  - `information_content`: Information content estimate
  - `redundancy`: Feature redundancy score

## 🎛️ Configuration Examples

### Maximum Dimensionality Configuration

```python
extractor = UltraHighDimExtractor(
    target_dimensions=40000,
    wavelets=['db4', 'db8', 'coif5', 'bior4.4', 'sym8'],
    max_dwt_levels=7,
    max_wpd_levels=6,
    feature_types=['statistical', 'energy', 'entropy', 'morphological', 'spectral'],
    optimize_for='image_reconstruction',
    n_jobs=-1
)
```

### Speed-Optimized Configuration

```python
extractor = UltraHighDimExtractor(
    target_dimensions=15000,
    wavelets=['db4', 'db8'],
    max_dwt_levels=4,
    max_wpd_levels=4,
    feature_types=['statistical', 'energy'],
    optimize_for='speed',
    n_jobs=-1
)
```

### Memory-Efficient Configuration

```python
extractor = UltraHighDimExtractor(
    target_dimensions=20000,
    wavelets=['db8', 'coif5'],
    max_dwt_levels=5,
    max_wpd_levels=4,
    batch_size=16,
    low_memory_mode=True,
    n_jobs=2
)
```

## 🔧 Advanced Usage

### Custom Wavelet Selection

```python
from core.base import WaveletAnalyzer

# Get available wavelets
available = WaveletAnalyzer.get_available_wavelets()
print("Available wavelet families:", list(available.keys()))

# Get EEG-optimized wavelets
recommended = WaveletAnalyzer.recommend_wavelets_for_eeg()
print("Recommended for EEG:", recommended)

# Create extractor with custom wavelets
extractor = UltraHighDimExtractor(
    wavelets=recommended[:4],  # Use first 4 recommended
    target_dimensions=30000
)
```

### Batch Processing

```python
# For large datasets
def process_large_dataset(data, batch_size=32):
    extractor = UltraHighDimExtractor(target_dimensions=25000)
    
    all_features = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        features = extractor.fit_transform(batch)
        all_features.append(features)
    
    return np.vstack(all_features)
```

### Performance Monitoring

```python
import time

# Monitor extraction performance
start_time = time.time()
features = extractor.fit_transform(data)
extraction_time = time.time() - start_time

print(f"Extracted {features.shape[1]:,} features in {extraction_time:.2f}s")
print(f"Speed: {features.shape[1]/extraction_time:.0f} features/second")

# Check memory usage
memory_est = extractor.get_memory_usage_estimate(len(data))
print(f"Memory usage: {memory_est['total_estimated']:.1f} MB")
```

## 🚨 Error Handling

### Common Exceptions

```python
try:
    features = extractor.fit_transform(data)
except ValueError as e:
    print(f"Data validation error: {e}")
except MemoryError as e:
    print(f"Memory error: {e}")
    # Try with smaller batch size or lower dimensions
except RuntimeError as e:
    print(f"Runtime error: {e}")
    # Check if waveletfeatures dependencies are available
```

### Debugging Tips

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.INFO)

# Check extractor configuration
info = extractor.get_extractor_info()
print(f"Number of sub-extractors: {len(info)}")
for i, extractor_info in enumerate(info):
    print(f"Extractor {i+1}: {extractor_info}")

# Validate data before processing
try:
    validated_data = validate_eeg_data(data)
    print("✅ Data validation passed")
except ValueError as e:
    print(f"❌ Data validation failed: {e}")
```

## 📊 Performance Guidelines

### Recommended Configurations by Use Case

| Use Case | Target Dims | Wavelets | Levels | Expected Speed |
|----------|-------------|----------|---------|----------------|
| Real-time BCI | 8,000 | ['db4'] | 3 | >10,000 feat/s |
| Image Reconstruction | 25,000 | ['db4','db8','coif5'] | 5 | >5,000 feat/s |
| Research Analysis | 35,000 | ['db4','db8','coif5','bior4.4'] | 6 | >3,000 feat/s |
| Maximum Quality | 45,000 | All available | 7 | >2,000 feat/s |

### Memory Requirements

| Samples | Features | Estimated Memory |
|---------|----------|------------------|
| 100 | 25,000 | ~35 MB |
| 500 | 25,000 | ~150 MB |
| 1,000 | 25,000 | ~300 MB |
| 1,000 | 40,000 | ~480 MB |
