"""
Setup script for UltraHighDimWaveletExtractor
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="UltraHighDimWaveletExtractor",
    version="1.0.0",
    author="EEG Research Team",
    author_email="research@eeg-team.com",
    description="Ultra-High Dimensional Wavelet Feature Extraction for EEG-to-Image Reconstruction",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/eeg-research/UltraHighDimWaveletExtractor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ]
    },
    keywords=[
        "eeg", "wavelet", "feature-extraction", "image-reconstruction", 
        "neuroscience", "brain-computer-interface", "signal-processing",
        "dwt", "wpd", "ultra-high-dimensional"
    ],
    project_urls={
        "Bug Reports": "https://github.com/eeg-research/UltraHighDimWaveletExtractor/issues",
        "Source": "https://github.com/eeg-research/UltraHighDimWaveletExtractor",
        "Documentation": "https://ultrahighdimwaveletextractor.readthedocs.io/",
    },
    include_package_data=True,
    zip_safe=False,
)
