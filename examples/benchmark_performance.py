#!/usr/bin/env python3
"""
Performance Benchmark for UltraHighDimWaveletExtractor
=====================================================

This script benchmarks the performance of UltraHighDimWaveletExtractor
across different configurations and data sizes.

Benchmarks:
- Feature extraction speed
- Memory usage
- Scalability analysis
- Quality vs speed trade-offs
- Comparison with standard methods
"""

import numpy as np
import time
import psutil
import sys
import os
from memory_profiler import profile
import matplotlib.pyplot as plt

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ultra_extractor import UltraHighDimExtractor
from core.preprocessing import create_optimal_preprocessor
from utils.validation import validate_eeg_data
from utils.metrics import FeatureQualityMetrics


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.results = {}
        self.memory_usage = {}
        
    def generate_test_data(self, n_samples, n_channels=14, n_timepoints=128):
        """Generate realistic test data."""
        np.random.seed(42)
        
        data = np.zeros((n_samples, n_channels, n_timepoints))
        t = np.linspace(0, 1, n_timepoints)
        
        for i in range(n_samples):
            for ch in range(n_channels):
                # Multi-frequency brain signals
                alpha = 2.0 * np.sin(2 * np.pi * 10 * t)
                beta = 1.5 * np.sin(2 * np.pi * 20 * t)
                gamma = 1.0 * np.sin(2 * np.pi * 40 * t)
                noise = 0.3 * np.random.randn(n_timepoints)
                data[i, ch, :] = alpha + beta + gamma + noise
        
        return data
    
    def benchmark_extraction_speed(self):
        """Benchmark feature extraction speed across different configurations."""
        print("🚀 BENCHMARKING EXTRACTION SPEED")
        print("-" * 50)
        
        # Test configurations
        configs = [
            {'name': 'Fast', 'target_dimensions': 10000, 'wavelets': ['db4'], 'levels': 3},
            {'name': 'Balanced', 'target_dimensions': 20000, 'wavelets': ['db4', 'db8'], 'levels': 4},
            {'name': 'High-Dim', 'target_dimensions': 30000, 'wavelets': ['db4', 'db8', 'coif5'], 'levels': 5},
            {'name': 'Ultra-High', 'target_dimensions': 40000, 'wavelets': ['db4', 'db8', 'coif5', 'bior4.4'], 'levels': 6}
        ]
        
        # Test data sizes
        data_sizes = [10, 25, 50, 100]
        
        results = {}
        
        for config in configs:
            config_name = config['name']
            results[config_name] = {}
            
            print(f"\n🔧 Testing {config_name} configuration...")
            
            # Create extractor
            extractor = UltraHighDimExtractor(
                target_dimensions=config['target_dimensions'],
                wavelets=config['wavelets'],
                max_dwt_levels=config['levels'],
                max_wpd_levels=config['levels']
            )
            
            for n_samples in data_sizes:
                print(f"   📊 {n_samples} samples...", end=' ')
                
                # Generate data
                data = self.generate_test_data(n_samples)
                
                # Preprocess
                preprocessor = create_optimal_preprocessor()
                clean_data = preprocessor.fit_transform(data)
                
                # Benchmark extraction
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                features = extractor.fit_transform(clean_data)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                extraction_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                results[config_name][n_samples] = {
                    'extraction_time': extraction_time,
                    'memory_used': memory_used,
                    'n_features': features.shape[1],
                    'features_per_second': features.shape[1] / extraction_time,
                    'samples_per_second': n_samples / extraction_time
                }
                
                print(f"{extraction_time:.2f}s, {features.shape[1]:,} features, {memory_used:.1f}MB")
        
        self.results['speed_benchmark'] = results
        return results
    
    def benchmark_scalability(self):
        """Benchmark scalability with increasing data sizes."""
        print("\n📈 BENCHMARKING SCALABILITY")
        print("-" * 50)
        
        # Large data sizes for scalability test
        data_sizes = [50, 100, 200, 500, 1000]
        
        # Use balanced configuration
        extractor = UltraHighDimExtractor(
            target_dimensions=25000,
            wavelets=['db4', 'db8'],
            max_dwt_levels=4,
            max_wpd_levels=4
        )
        
        results = {}
        
        for n_samples in data_sizes:
            print(f"🔧 Testing {n_samples} samples...")
            
            # Generate data
            data = self.generate_test_data(n_samples)
            
            # Preprocess
            preprocessor = create_optimal_preprocessor()
            clean_data = preprocessor.fit_transform(data)
            
            # Benchmark
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            features = extractor.fit_transform(clean_data)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            results[n_samples] = {
                'extraction_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'n_features': features.shape[1],
                'time_per_sample': (end_time - start_time) / n_samples,
                'memory_per_sample': (end_memory - start_memory) / n_samples
            }
            
            print(f"   ⏱️  {results[n_samples]['extraction_time']:.2f}s total")
            print(f"   📊 {results[n_samples]['time_per_sample']:.3f}s per sample")
            print(f"   💾 {results[n_samples]['memory_used']:.1f}MB total")
        
        self.results['scalability'] = results
        return results
    
    def benchmark_quality_vs_speed(self):
        """Benchmark quality vs speed trade-offs."""
        print("\n⚖️  BENCHMARKING QUALITY VS SPEED")
        print("-" * 50)
        
        # Different quality/speed configurations
        configs = [
            {'name': 'Speed-Optimized', 'target_dimensions': 8000, 'wavelets': ['db4'], 'levels': 3},
            {'name': 'Balanced', 'target_dimensions': 15000, 'wavelets': ['db4', 'db8'], 'levels': 4},
            {'name': 'Quality-Optimized', 'target_dimensions': 30000, 'wavelets': ['db4', 'db8', 'coif5'], 'levels': 5},
            {'name': 'Maximum-Quality', 'target_dimensions': 45000, 'wavelets': ['db4', 'db8', 'coif5', 'bior4.4'], 'levels': 6}
        ]
        
        # Test data
        data = self.generate_test_data(50)
        preprocessor = create_optimal_preprocessor()
        clean_data = preprocessor.fit_transform(data)
        
        results = {}
        
        for config in configs:
            config_name = config['name']
            print(f"\n🔧 Testing {config_name}...")
            
            # Create extractor
            extractor = UltraHighDimExtractor(
                target_dimensions=config['target_dimensions'],
                wavelets=config['wavelets'],
                max_dwt_levels=config['levels'],
                max_wpd_levels=config['levels']
            )
            
            # Benchmark
            start_time = time.time()
            features = extractor.fit_transform(clean_data)
            extraction_time = time.time() - start_time
            
            # Assess quality
            quality_metrics = FeatureQualityMetrics.compute_basic_metrics(features)
            
            results[config_name] = {
                'extraction_time': extraction_time,
                'n_features': features.shape[1],
                'features_per_second': features.shape[1] / extraction_time,
                'quality_score': np.mean([v for v in quality_metrics.values() if isinstance(v, (int, float))]),
                'snr_estimate': quality_metrics.get('snr_estimate', 0),
                'information_content': quality_metrics.get('information_content', 0)
            }
            
            print(f"   ⏱️  {extraction_time:.2f}s")
            print(f"   📊 {features.shape[1]:,} features")
            print(f"   🎯 Quality score: {results[config_name]['quality_score']:.3f}")
        
        self.results['quality_vs_speed'] = results
        return results
    
    def compare_with_standard_methods(self):
        """Compare with standard wavelet feature extraction methods."""
        print("\n🔬 COMPARING WITH STANDARD METHODS")
        print("-" * 50)
        
        # Test data
        data = self.generate_test_data(50)
        preprocessor = create_optimal_preprocessor()
        clean_data = preprocessor.fit_transform(data)
        
        # Import standard extractors
        try:
            sys.path.append('../../waveletfeatures')
            from dwt_features import DWTFeatureExtractor
            from wpd_features import WPDFeatureExtractor
            
            methods = {
                'Standard DWT': DWTFeatureExtractor(wavelet='db4', levels=4),
                'Deep DWT': DWTFeatureExtractor(wavelet='db8', levels=6),
                'Standard WPD': WPDFeatureExtractor(wavelet='db4', levels=3),
                'Deep WPD': WPDFeatureExtractor(wavelet='db8', levels=5),
                'Ultra-High Dim': UltraHighDimExtractor(target_dimensions=25000)
            }
            
            results = {}
            
            for name, extractor in methods.items():
                print(f"🔧 Testing {name}...")
                
                start_time = time.time()
                features = extractor.fit_transform(clean_data)
                extraction_time = time.time() - start_time
                
                results[name] = {
                    'n_features': features.shape[1],
                    'extraction_time': extraction_time,
                    'features_per_second': features.shape[1] / extraction_time,
                    'dimensionality_ratio': features.shape[1] / 18000  # vs fMRI visual cortex
                }
                
                print(f"   📊 {features.shape[1]:,} features in {extraction_time:.2f}s")
                print(f"   🧠 {results[name]['dimensionality_ratio']:.1%} of fMRI visual cortex")
            
            self.results['method_comparison'] = results
            return results
            
        except ImportError:
            print("   ⚠️  Standard extractors not available for comparison")
            return {}
    
    def create_performance_plots(self):
        """Create performance visualization plots."""
        print("\n📊 CREATING PERFORMANCE PLOTS")
        print("-" * 50)
        
        # Speed benchmark plot
        if 'speed_benchmark' in self.results:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Extraction time vs data size
            plt.subplot(2, 3, 1)
            for config_name, config_results in self.results['speed_benchmark'].items():
                sizes = list(config_results.keys())
                times = [config_results[size]['extraction_time'] for size in sizes]
                plt.plot(sizes, times, marker='o', label=config_name)
            plt.xlabel('Number of Samples')
            plt.ylabel('Extraction Time (s)')
            plt.title('Extraction Time vs Data Size')
            plt.legend()
            plt.grid(True)
            
            # Plot 2: Features per second
            plt.subplot(2, 3, 2)
            for config_name, config_results in self.results['speed_benchmark'].items():
                sizes = list(config_results.keys())
                fps = [config_results[size]['features_per_second'] for size in sizes]
                plt.plot(sizes, fps, marker='s', label=config_name)
            plt.xlabel('Number of Samples')
            plt.ylabel('Features per Second')
            plt.title('Feature Extraction Speed')
            plt.legend()
            plt.grid(True)
            
            # Plot 3: Memory usage
            plt.subplot(2, 3, 3)
            for config_name, config_results in self.results['speed_benchmark'].items():
                sizes = list(config_results.keys())
                memory = [config_results[size]['memory_used'] for size in sizes]
                plt.plot(sizes, memory, marker='^', label=config_name)
            plt.xlabel('Number of Samples')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage vs Data Size')
            plt.legend()
            plt.grid(True)
        
        # Scalability plot
        if 'scalability' in self.results:
            plt.subplot(2, 3, 4)
            sizes = list(self.results['scalability'].keys())
            times_per_sample = [self.results['scalability'][size]['time_per_sample'] for size in sizes]
            plt.plot(sizes, times_per_sample, marker='o', color='red')
            plt.xlabel('Number of Samples')
            plt.ylabel('Time per Sample (s)')
            plt.title('Scalability Analysis')
            plt.grid(True)
        
        # Quality vs Speed
        if 'quality_vs_speed' in self.results:
            plt.subplot(2, 3, 5)
            configs = list(self.results['quality_vs_speed'].keys())
            speeds = [self.results['quality_vs_speed'][config]['features_per_second'] for config in configs]
            qualities = [self.results['quality_vs_speed'][config]['quality_score'] for config in configs]
            plt.scatter(speeds, qualities, s=100)
            for i, config in enumerate(configs):
                plt.annotate(config, (speeds[i], qualities[i]), xytext=(5, 5), textcoords='offset points')
            plt.xlabel('Features per Second')
            plt.ylabel('Quality Score')
            plt.title('Quality vs Speed Trade-off')
            plt.grid(True)
        
        # Method comparison
        if 'method_comparison' in self.results:
            plt.subplot(2, 3, 6)
            methods = list(self.results['method_comparison'].keys())
            n_features = [self.results['method_comparison'][method]['n_features'] for method in methods]
            plt.bar(range(len(methods)), n_features)
            plt.xticks(range(len(methods)), methods, rotation=45)
            plt.ylabel('Number of Features')
            plt.title('Feature Count Comparison')
            plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('performance_benchmark.png', dpi=150, bbox_inches='tight')
        print("   ✅ Performance plots saved as 'performance_benchmark.png'")
    
    def generate_report(self):
        """Generate comprehensive performance report."""
        print("\n📋 GENERATING PERFORMANCE REPORT")
        print("=" * 60)
        
        report = []
        report.append("# UltraHighDimWaveletExtractor Performance Report\n")
        
        # Speed benchmark summary
        if 'speed_benchmark' in self.results:
            report.append("## Speed Benchmark Results\n")
            for config_name, config_results in self.results['speed_benchmark'].items():
                avg_fps = np.mean([r['features_per_second'] for r in config_results.values()])
                avg_memory = np.mean([r['memory_used'] for r in config_results.values()])
                max_features = max([r['n_features'] for r in config_results.values()])
                
                report.append(f"### {config_name} Configuration")
                report.append(f"- Average speed: {avg_fps:.0f} features/second")
                report.append(f"- Average memory: {avg_memory:.1f} MB")
                report.append(f"- Maximum features: {max_features:,}")
                report.append("")
        
        # Quality vs Speed summary
        if 'quality_vs_speed' in self.results:
            report.append("## Quality vs Speed Analysis\n")
            best_balance = max(self.results['quality_vs_speed'].items(), 
                             key=lambda x: x[1]['quality_score'] / x[1]['extraction_time'])
            report.append(f"**Best balanced configuration:** {best_balance[0]}")
            report.append(f"- Quality score: {best_balance[1]['quality_score']:.3f}")
            report.append(f"- Speed: {best_balance[1]['features_per_second']:.0f} features/second")
            report.append("")
        
        # Method comparison summary
        if 'method_comparison' in self.results:
            report.append("## Comparison with Standard Methods\n")
            ultra_high = self.results['method_comparison'].get('Ultra-High Dim', {})
            if ultra_high:
                report.append(f"**Ultra-High Dim Extractor:**")
                report.append(f"- Features: {ultra_high['n_features']:,}")
                report.append(f"- Speed: {ultra_high['features_per_second']:.0f} features/second")
                report.append(f"- fMRI ratio: {ultra_high['dimensionality_ratio']:.1%}")
                report.append("")
        
        # Save report
        with open('performance_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        print("   ✅ Performance report saved as 'performance_report.md'")
        
        # Print summary to console
        print("\n🎯 PERFORMANCE SUMMARY:")
        if 'speed_benchmark' in self.results:
            ultra_config = self.results['speed_benchmark'].get('Ultra-High', {})
            if ultra_config:
                sample_result = list(ultra_config.values())[0]
                print(f"   🚀 Ultra-High config: {sample_result['n_features']:,} features")
                print(f"   ⚡ Speed: {sample_result['features_per_second']:.0f} features/second")
        
        print("   ✅ All benchmarks completed successfully!")


def main():
    """Main benchmark function."""
    print("⚡ ULTRAHIGHDIMWAVELETEXTRACTOR PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    benchmark = PerformanceBenchmark()
    
    # Run all benchmarks
    benchmark.benchmark_extraction_speed()
    benchmark.benchmark_scalability()
    benchmark.benchmark_quality_vs_speed()
    benchmark.compare_with_standard_methods()
    
    # Create visualizations and report
    benchmark.create_performance_plots()
    benchmark.generate_report()
    
    print("\n🏁 BENCHMARK COMPLETED!")
    print("📊 Check 'performance_benchmark.png' for visualizations")
    print("📋 Check 'performance_report.md' for detailed report")


if __name__ == "__main__":
    main()
