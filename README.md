# NIST OCR

Handwritten character recognition using classical computer vision and deep learning methods on the NIST SD19 dataset.

## Dataset

Download from: https://www.nist.gov/srd/nist-special-database-19

Extract so that `by_class/` folder is at root level.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run with default settings (all 5 methods)
python benchmark.py

# Skip learning curves for faster run
python benchmark.py --skip-learning-curves

# Train on all 62 classes (A-Z, a-z, 0-9)
python benchmark.py --config config_62_classes.yaml
```

## Usage

### Using Configuration Files

```bash
# Use default config
python benchmark.py

# Use 62-class config
python benchmark.py --config config_62_classes.yaml
```

### Train Specific Models

```bash
# Only CNN (both with and without augmentation)
python benchmark.py --skip-hog --skip-zernike --skip-projection

# Only classical methods
python benchmark.py --skip-cnn
```

### Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--config PATH` | YAML config file |
| `--train-limit N` | Number of training samples |
| `--test-limit N` | Number of test samples |
| `--skip-hog` | Skip HOG+SVM |
| `--skip-cnn` | Skip CNN models |
| `--skip-zernike` | Skip Zernike+SVM |
| `--skip-projection` | Skip Projection+kNN |
| `--skip-learning-curves` | Skip learning curves |
| `--no-visualizations` | Disable visualizations |

## Implemented Methods

### Classical Methods

| Method | Features | Classifier |
|--------|----------|------------|
| HOG + SVM | Histogram of Oriented Gradients | Linear SVM |
| Zernike + SVM | Zernike Moments | RBF SVM |
| Projection + kNN | Row/Column Projections | k-Nearest Neighbors |

### Deep Learning

| Model | Description |
|-------|-------------|
| CNN | 4-layer CNN with BatchNorm, AdamW optimizer |
| CNN + Aug | Same architecture with data augmentation |
| ResNet | Small ResNet-style CNN adapted for grayscale OCR |
| ResNet + Aug | ResNet with data augmentation |

Data augmentation includes: rotation, elastic distortion, translation, scaling, and noise.

## Output

The benchmark produces:
- **Summary table** with accuracy, training time, and inference latency
- **Learning curves** for each method (train/test accuracy vs dataset size)
- **Confusion matrices** per model
- **Per-class metrics** (precision, recall, F1)
- **Method comparison dashboard** visualizing all results

## Data Splitting

| Split | Source | Purpose |
|-------|--------|---------|
| Training | `train_*` partitions (85%) | Model training |
| Validation | From training (15%) | Early stopping (CNN) |
| Testing | `hsf_7` partition | Final evaluation |

## Project Structure

```
nist_ocr/
├── benchmark.py              # Main training pipeline
├── dataset.py                # Dataset loading
├── cnn.py                    # CNN model
├── config.py                 # Configuration management
├── visualizations.py         # All plots and metrics
├── classical_methods/
│   ├── base_classical.py     # Base class
│   ├── hog_ocr.py            # HOG + SVM
│   ├── zernike_ocr.py        # Zernike + SVM
│   └── projection_ocr.py     # Projection + kNN
├── preprocessing/
│   ├── image_preprocessing.py
│   └── augmentation.py
├── default_config.yaml       # Default config
└── config_62_classes.yaml    # Full 62-class config
```

## Features

- **Quantitative comparison** between classical and deep learning methods
- **Class balancing** with weighted loss/classifiers
- **Early stopping** when validation plateaus
- **Model checkpoints** saves best models during training
- **Learning rate scheduling** with ReduceLROnPlateau
- **Learning curves** for all methods
- **Confusion matrix** visualization
- **Per-class metrics** (precision, recall, F1)
- **Comparison dashboard** with accuracy, training time, latency plots
