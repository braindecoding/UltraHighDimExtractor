# UltraHighDimWaveletExtractor Usage Guide

## 🎯 Overview

UltraHighDimWaveletExtractor is designed for **ultra-high dimensional feature extraction** from EEG signals, specifically optimized for **image reconstruction tasks**. It can extract **35,000+ features** that compete with fMRI visual cortex dimensionality.

## 🚀 Quick Start

### Basic Usage

```python
import numpy as np
import sys
sys.path.append('path/to/UltraHighDimWaveletExtractor')

from core.ultra_extractor import UltraHighDimExtractor
from core.preprocessing import create_optimal_preprocessor
from utils.validation import validate_eeg_data

# 1. Load your EEG data
# Shape: (n_samples, n_channels, n_timepoints)
eeg_data = np.load('your_eeg_data.npy')

# 2. Validate data
validated_data = validate_eeg_data(eeg_data)

# 3. Preprocess for image reconstruction
preprocessor = create_optimal_preprocessor(task_type='image_reconstruction')
clean_data = preprocessor.fit_transform(validated_data)

# 4. Extract ultra-high dimensional features
extractor = UltraHighDimExtractor(target_dimensions=25000)
features = extractor.fit_transform(clean_data)

print(f"Extracted {features.shape[1]:,} features from {features.shape[0]} samples")
```

### Advanced Configuration

```python
# Custom configuration for maximum dimensionality
extractor = UltraHighDimExtractor(
    target_dimensions=40000,           # Target feature count
    wavelets=['db4', 'db8', 'coif5'],  # Multiple wavelets
    max_dwt_levels=6,                  # Deep DWT decomposition
    max_wpd_levels=5,                  # Deep WPD decomposition
    feature_types=['statistical', 'energy', 'entropy', 'morphological'],
    sampling_rate=128,                 # Your EEG sampling rate
    n_jobs=4                          # Parallel processing
)
```

## 📊 Feature Extraction Strategies

### 1. Standard Approach (10K-15K features)
```python
extractor = UltraHighDimExtractor(
    target_dimensions=15000,
    wavelets=['db4', 'db8'],
    max_dwt_levels=4,
    max_wpd_levels=4
)
```

### 2. Maximum Dimensionality (35K+ features)
```python
extractor = UltraHighDimExtractor(
    target_dimensions=40000,
    wavelets=['db4', 'db8', 'coif5', 'bior4.4'],
    max_dwt_levels=6,
    max_wpd_levels=5,
    include_cross_frequency=True,
    include_time_frequency=True
)
```

### 3. Balanced Performance (20K features)
```python
extractor = UltraHighDimExtractor(
    target_dimensions=20000,
    wavelets=['db8', 'coif5'],
    max_dwt_levels=5,
    max_wpd_levels=4,
    optimize_for='speed'
)
```

## 🧠 Image Reconstruction Pipeline

### Complete Workflow

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Prepare data
def prepare_eeg_image_data(eeg_data, image_data):
    """Prepare EEG and image data for reconstruction."""
    
    # Validate EEG data
    eeg_validated = validate_eeg_data(eeg_data)
    
    # Preprocess EEG
    preprocessor = create_optimal_preprocessor(
        task_type='image_reconstruction',
        preserve_high_freq=True
    )
    eeg_clean = preprocessor.fit_transform(eeg_validated)
    
    # Extract ultra-high dim features
    extractor = UltraHighDimExtractor(target_dimensions=30000)
    eeg_features = extractor.fit_transform(eeg_clean)
    
    # Flatten images
    image_flat = image_data.reshape(image_data.shape[0], -1)
    
    return eeg_features, image_flat

# 2. Train reconstruction model
def train_reconstruction_model(eeg_features, images):
    """Train image reconstruction model."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        eeg_features, images, test_size=0.2, random_state=42
    )
    
    # Use Ridge regression for reconstruction
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Reconstruction MSE: {mse:.4f}")
    
    return model, X_test, y_test, y_pred

# 3. Example usage
eeg_data = np.random.randn(100, 14, 128)  # Your EEG data
image_data = np.random.randn(100, 64, 64)  # Your image data

eeg_features, images_flat = prepare_eeg_image_data(eeg_data, image_data)
model, X_test, y_test, y_pred = train_reconstruction_model(eeg_features, images_flat)
```

## 🔬 Performance Optimization

### Memory Management

```python
# For large datasets
extractor = UltraHighDimExtractor(
    target_dimensions=25000,
    batch_size=32,              # Process in batches
    low_memory_mode=True,       # Reduce memory usage
    cache_wavelets=False        # Don't cache wavelet transforms
)

# Check memory requirements
memory_est = extractor.get_memory_usage_estimate(n_samples=1000)
print(f"Estimated memory usage: {memory_est['total_estimated']:.1f} MB")
```

### Speed Optimization

```python
# For faster processing
extractor = UltraHighDimExtractor(
    target_dimensions=20000,
    n_jobs=-1,                  # Use all CPU cores
    optimize_for='speed',       # Speed over dimensionality
    skip_redundant_features=True
)

# Estimate processing time
time_est = extractor.estimate_extraction_time(n_samples=500)
print(f"Estimated processing time: {time_est:.1f} seconds")
```

## 📈 Quality Assessment

### Feature Quality Metrics

```python
from utils.metrics import FeatureQualityMetrics

# Extract features
features = extractor.fit_transform(clean_data)

# Compute quality metrics
metrics = FeatureQualityMetrics.compute_comprehensive_metrics(features)

print("Feature Quality Report:")
print(f"  Signal-to-Noise Ratio: {metrics['snr']:.2f}")
print(f"  Feature Stability: {metrics['stability']:.2f}")
print(f"  Information Content: {metrics['information_content']:.2f}")
print(f"  Redundancy Score: {metrics['redundancy']:.2f}")
```

### Reconstruction Quality

```python
def evaluate_reconstruction_quality(original_images, reconstructed_images):
    """Evaluate image reconstruction quality."""
    
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    
    ssim_scores = []
    psnr_scores = []
    
    for orig, recon in zip(original_images, reconstructed_images):
        # Reshape to image format
        orig_img = orig.reshape(64, 64)
        recon_img = recon.reshape(64, 64)
        
        # Compute metrics
        ssim_score = ssim(orig_img, recon_img, data_range=1.0)
        psnr_score = psnr(orig_img, recon_img, data_range=1.0)
        
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
    
    return {
        'mean_ssim': np.mean(ssim_scores),
        'mean_psnr': np.mean(psnr_scores),
        'std_ssim': np.std(ssim_scores),
        'std_psnr': np.std(psnr_scores)
    }
```

## 🎛️ Configuration Options

### Wavelet Selection

```python
# For different signal characteristics
configs = {
    'smooth_signals': {
        'wavelets': ['db8', 'coif5'],
        'focus': 'low_frequency'
    },
    'sharp_transients': {
        'wavelets': ['db4', 'bior4.4'],
        'focus': 'high_frequency'
    },
    'mixed_content': {
        'wavelets': ['db4', 'db8', 'coif5', 'bior4.4'],
        'focus': 'full_spectrum'
    }
}
```

### Task-Specific Optimization

```python
# Image reconstruction
extractor_img = UltraHighDimExtractor(
    target_dimensions=30000,
    optimize_for='image_reconstruction',
    preserve_spatial_info=True,
    enhance_edge_features=True
)

# Classification
extractor_cls = UltraHighDimExtractor(
    target_dimensions=15000,
    optimize_for='classification',
    focus_discriminative=True,
    reduce_noise_features=True
)

# General analysis
extractor_gen = UltraHighDimExtractor(
    target_dimensions=25000,
    optimize_for='general',
    balanced_features=True
)
```

## 🚨 Troubleshooting

### Common Issues

1. **Memory Errors**
   ```python
   # Reduce batch size or target dimensions
   extractor = UltraHighDimExtractor(
       target_dimensions=15000,  # Reduce from 30000
       batch_size=16,            # Reduce from 32
       low_memory_mode=True
   )
   ```

2. **Slow Processing**
   ```python
   # Enable parallel processing
   extractor = UltraHighDimExtractor(
       n_jobs=-1,                # Use all cores
       optimize_for='speed',
       skip_redundant_features=True
   )
   ```

3. **Poor Reconstruction Quality**
   ```python
   # Increase dimensionality and preprocessing
   preprocessor = create_optimal_preprocessor(
       task_type='image_reconstruction',
       aggressive_artifact_removal=True,
       preserve_high_freq=True
   )
   
   extractor = UltraHighDimExtractor(
       target_dimensions=35000,  # Increase dimensions
       include_cross_frequency=True,
       enhance_edge_features=True
   )
   ```

## 📚 Next Steps

1. **Advanced Examples**: See `examples/` folder for complete implementations
   - `image_reconstruction_example.py`: Complete EEG-to-image reconstruction
   - `benchmark_performance.py`: Performance analysis and optimization
   - `basic_usage.py`: Simple feature extraction examples

2. **API Reference**: Check `docs/API_REFERENCE.md` for detailed function documentation

3. **Performance Benchmarks**: Run `examples/benchmark_performance.py`
   ```bash
   cd examples/
   python benchmark_performance.py
   ```

4. **Testing**: Run comprehensive tests
   ```bash
   python final_test.py
   python test_ultra_extractor.py
   ```

## 🎯 Real-World Applications

### EEG-to-Image Reconstruction
- **Visual imagery decoding**: Reconstruct images from visual imagination
- **Dream visualization**: Decode visual content from sleep EEG
- **BCI applications**: Brain-computer interfaces for image generation

### Neuroscience Research
- **Visual cortex analysis**: Compare EEG features with fMRI data
- **Cognitive state classification**: Identify mental states from EEG
- **Attention mechanisms**: Decode visual attention patterns

### Clinical Applications
- **Neurological assessment**: Quantify visual processing deficits
- **Rehabilitation**: Monitor recovery of visual functions
- **Diagnostic tools**: Early detection of visual processing disorders
