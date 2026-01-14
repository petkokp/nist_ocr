# NIST OCR

## Dataset

Download dataset from https://s3.amazonaws.com/nist-srd/SD19/by_class.zip (the `by_class` folder should be at root level)

## Install dependencies

```
pip install -r requirements.txt
```

## Benchmark

```
python benchmark.py
```

Expected result:

| Model | Accuracy | Train Time | Latency |
| :--- | :--- | :--- | :--- |
| HOG + SVM | 98.20% | 2.28 | 1.08 |
| Simple CNN | 98.40% | 2.89 | 0.10 |

## File structure

- `dataset.py` - preprocess and load dataset, prepare it for experiments
- `hog.py` - implementation of the histogram of oriented gradients method
- `cnn.py` - implementation of simple convolutional neural network
- `benchmark.py` - run training/inference and compare the different methods on a test subset of the dataset