# Comprehensive Emotion Classification Model Comparison

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.0+-yellow.svg)](https://huggingface.co/transformers/)

A systematic evaluation framework comparing state-of-the-art deep learning architectures for text-based emotion classification.

## Overview

This repository presents a comprehensive benchmarking study of **9+ deep learning models** for emotion classification, ranging from traditional neural networks to modern transformer-based architectures. The framework enables fair comparison across different model paradigms with standardized evaluation metrics and professional-grade visualizations.

## Evaluated Models

### Traditional Architectures
- **BiLSTM with Attention**: Bidirectional LSTM enhanced with attention mechanism
- **CNN**: Multi-filter convolutional neural network for text classification

### Transformer-Based Models
- **RoBERTa** (Base/Large): Robustly optimized BERT pretraining approach
- **DeBERTa-v3**: Enhanced BERT with disentangled attention
- **DistilBERT**: Lightweight, distilled version of BERT
- **ELECTRA**: Efficiently learning encoder through replaced token detection
- **XLNet**: Generalized autoregressive pretraining
- **ALBERT**: A lite BERT with parameter sharing

## Key Features

- 🔧 **Unified Framework**: Modular architecture supporting both traditional and transformer models
- 📊 **Comprehensive Metrics**: Accuracy, F1-score, precision, recall with per-class analysis
- ⚡ **Efficiency Analysis**: Training time, inference speed, and model size comparison
- 🎨 **Interactive Visualizations**: Professional-grade charts and analysis plots
- 🎯 **Class Balance Handling**: Weighted loss functions for imbalanced datasets
- 🔄 **Reproducible Pipeline**: Standardized data processing and evaluation protocols

## Quick Start

```python
# Initialize the comparison framework
from emotion_classifier_comparison import run_quick_test, run_full_comparison

# Quick test with selected models
results = run_quick_test()

# Full comprehensive comparison
FULL_CONFIG['run_full_comparison'] = True
full_results = run_full_comparison()

# Custom model selection
custom_results = run_custom_comparison(['roberta-base', 'distilbert-base', 'bilstm-attention'])
```

## Evaluation Protocol

- **Dataset**: DAIR-AI Emotion Classification (6 classes: sadness, joy, love, anger, fear, surprise)
- **Split**: 70% training, 15% validation, 15% testing
- **Metrics**: Macro/weighted F1-score, accuracy, per-class performance
- **Hardware**: CUDA-enabled GPU support with automatic fallback to CPU

## Results Structure

```
results/
├── model_comparison.csv          # Quantitative comparison table
├── full_comparison_results.json  # Detailed metrics and confusion matrices
└── evaluation_results.json       # Per-model performance data

visualizations/
├── performance_comparison.html    # Interactive performance charts
├── confusion_matrices.html       # Multi-model confusion matrices
├── training_curves.html         # Learning progression analysis
├── radar_chart.html             # Per-emotion performance radar
└── efficiency_analysis.html     # Speed vs accuracy trade-offs
```

## Dependencies

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn plotly tqdm
```