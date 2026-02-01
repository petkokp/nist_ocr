# NIST OCR Benchmark

A comprehensive benchmarking framework for comparing classical computer vision and deep learning methods on handwritten character recognition using the NIST Special Database 19.

## Features

- **8 OCR methods** - 4 classical + 4 deep learning approaches
- **Comprehensive metrics** - Accuracy, precision, recall, F1, confusion matrices
- **Data augmentation** - Rotation, elastic distortion, translation, scaling, noise
- **Preprocessing pipeline** - Adaptive thresholding, morphology, deskewing
- **Class balancing** - Weighted loss functions and classifiers
- **Early stopping** - Prevents overfitting in deep learning models
- **Model checkpoints** - Saves best models during training
- **Configurable** - YAML configs + command-line overrides

## Implemented Methods

### Classical Methods

| Method | Features | Classifier | Description |
|--------|----------|------------|-------------|
| LBP + SVM | Local Binary Patterns | Linear SVM | Texture-based features |
| HOG + SVM | Histogram of Oriented Gradients | Linear SVM | Edge orientation histograms |
| Zernike + SVM | Zernike Moments | RBF SVM | Rotation-invariant shape descriptors |
| Projection + kNN | Row/Column Projections | k-Nearest Neighbors | Simple projection histograms |

### Deep Learning Methods

| Model | Architecture | Description |
|-------|--------------|-------------|
| CNN | 4-layer ConvNet | BatchNorm, Dropout, AdamW optimizer |
| CNN + Aug | 4-layer ConvNet | With data augmentation |
| ResNet | ResNet-style | Residual connections, adapted for grayscale |
| ResNet + Aug | ResNet-style | With data augmentation |

## Installation

### Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd nist_ocr

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| torch, torchvision | Deep learning framework |
| scikit-learn | Classical ML classifiers |
| scikit-image | Image processing, LBP, HOG |
| mahotas | Zernike moments |
| opencv-python | Image preprocessing |
| matplotlib, seaborn | Visualization |

## Dataset

### Download

1. Download NIST Special Database 19 from: https://www.nist.gov/srd/nist-special-database-19
2. Extract the archive so that `by_class/` folder is at the project root

### Structure

```
nist_ocr/
├── by_class/
│   ├── 30/          # Character '0'
│   ├── 31/          # Character '1'
│   ├── ...
│   ├── 41/          # Character 'A'
│   ├── ...
│   └── 7a/          # Character 'z'
```

### Data Splits

| Split | Partitions | Purpose |
|-------|------------|---------|
| Training | `train_30`, `train_31`, `train_32` | Model training (85%) |
| Validation | From training set | Early stopping (15%) |
| Test | `hsf_7`, `hsf_4`, `hsf_0` | Final evaluation |

## Quick Start

```bash
# Run with default settings (all 8 methods)
python benchmark.py

# Fast run - skip learning curves
python benchmark.py --skip-learning-curves

# Minimal run - no visualizations
python benchmark.py --skip-learning-curves --no-visualizations

# Full 62-class training (A-Z, a-z, 0-9)
python benchmark.py --config config_62_classes.yaml
```

## Configuration

### Using Config Files

```bash
# Default config (auto-loaded)
python benchmark.py

# Custom config
python benchmark.py --config config_62_classes.yaml
```

### Config File Structure

```yaml
dataset:
  root: "."
  train_limit: 2000        # Samples for training
  test_limit: 500          # Samples for testing
  image_size: 32           # Resize images to NxN

cnn:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adamw"       # adamw, adam, sgd
  early_stopping_patience: 5

augmentation:
  rotation_range: 15       # Degrees
  elastic_alpha: 34
  translation_range: 0.1
  scale_range: [0.9, 1.1]
  noise_std: 0.05

experiment:
  skip_learning_curves: false
  skip_cnn: false
  skip_resnet: false
  results_dir: "results"
```

See [default_config.yaml](default_config.yaml) for all options.

## Command-Line Arguments

### Dataset Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-root PATH` | Root directory with `by_class/` | `.` |
| `--train-limit N` | Number of training samples | 2000 |
| `--test-limit N` | Number of test samples | 500 |
| `--image-size N` | Resize images to NxN | 32 |

### Model Selection

| Argument | Description |
|----------|-------------|
| `--skip-lbp` | Skip LBP + SVM |
| `--skip-hog` | Skip HOG + SVM |
| `--skip-zernike` | Skip Zernike + SVM |
| `--skip-projection` | Skip Projection + kNN |
| `--skip-cnn` | Skip CNN models |
| `--skip-resnet` | Skip ResNet models |

### Training Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--cnn-epochs N` | Training epochs for CNN/ResNet | 10 |
| `--batch-size N` | Batch size | 32 |
| `--learning-rate F` | Learning rate | 0.001 |
| `--optimizer TYPE` | adam, adamw, sgd | adamw |
| `--early-stopping-patience N` | Epochs before early stop | 5 |

### Experiment Control

| Argument | Description |
|----------|-------------|
| `--skip-learning-curves` | Skip learning curve generation |
| `--no-visualizations` | Disable all visualizations |
| `--results-dir PATH` | Output directory for results |
| `--checkpoint-dir PATH` | Directory for model checkpoints |

### Preprocessing

| Argument | Description |
|----------|-------------|
| `--preprocessing` | Enable preprocessing for classical methods |
| `--threshold-method TYPE` | otsu or sauvola |
| `--no-deskew` | Disable deskewing |

## Output

Results are saved to `results/<model_name>/`:

```
results/
├── CNN/
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   ├── misclassifications.png
│   ├── training_history.png
│   └── predictions.csv
├── HOG_+_SVM/
│   └── ...
├── method_comparison.png      # All methods compared
└── ...
```

### Generated Visualizations

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Prediction error heatmap |
| `per_class_metrics.png` | Precision/Recall/F1 per character |
| `misclassifications.png` | Sample incorrect predictions |
| `training_history.png` | Loss and accuracy curves (CNN only) |
| `predictions.csv` | All predictions with confidence scores |
| `method_comparison.png` | Bar charts comparing all methods |

### Summary Table

The benchmark outputs a summary table:

```
Model                | Accuracy   | Train Time   | Latency
----------------------------------------------------------
LBP + SVM            | 16.70%     | 52.9         | 0.59
HOG + SVM            | 63.30%     | 55.2         | 0.17
Zernike + SVM        | 18.00%     | 857.2        | 10.58
Projection + kNN     | 62.90%     | 0.6          | 0.15
CNN + Aug            | 75.50%     | 361.9        | 0.01
CNN                  | 77.20%     | 30.9         | 0.01
ResNet + Aug         | 80.00%     | 772.7        | 0.08
ResNet               | 81.90%     | 352.8        | 0.08
```

## Project Structure

```
nist_ocr/
├── benchmark.py                 # Main entry point
├── dataset.py                   # NIST dataset loader
├── config.py                    # Configuration management
├── visualizations.py            # Plotting and metrics
│
├── classical_methods/
│   ├── __init__.py
│   ├── base_classical.py        # Base class for classical methods
│   ├── lbp_ocr.py               # LBP + SVM
│   ├── hog_ocr.py               # HOG + SVM
│   ├── zernike_ocr.py           # Zernike Moments + SVM
│   └── projection_ocr.py        # Projection + kNN
│
├── deep_learning_methods/
│   ├── cnn.py                   # Custom CNN
│   └── resnet_ocr.py            # ResNet architecture
│
├── preprocessing/
│   ├── __init__.py
│   ├── image_preprocessing.py   # Thresholding, morphology, deskew
│   └── augmentation.py          # Data augmentation pipeline
│
├── default_config.yaml          # Default configuration
├── config_62_classes.yaml       # Full 62-class configuration
├── requirements.txt             # Python dependencies
└── README.md
```

## Examples

### Train only classical methods

```bash
python benchmark.py --skip-cnn --skip-resnet --skip-learning-curves
```

### Train only deep learning methods

```bash
python benchmark.py --skip-lbp --skip-hog --skip-zernike --skip-projection
```

### Quick test with small dataset

```bash
python benchmark.py --train-limit 500 --test-limit 100 --skip-learning-curves
```

### Full training on 62 classes

```bash
python benchmark.py --config config_62_classes.yaml
```

### Custom CNN hyperparameters

```bash
python benchmark.py \
  --skip-hog --skip-zernike --skip-projection --skip-lbp --skip-resnet \
  --cnn-epochs 20 \
  --batch-size 64 \
  --learning-rate 0.0005 \
  --optimizer adamw
```

## License

This project is for educational and research purposes.

## References

- NIST Special Database 19: https://www.nist.gov/srd/nist-special-database-19
- Local Binary Patterns: Ojala et al. (2002)
- Histogram of Oriented Gradients: Dalal & Triggs (2005)
- Zernike Moments: Khotanzad & Hong (1990)
