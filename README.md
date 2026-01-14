# NIST OCR

Download dataset from https://s3.amazonaws.com/nist-srd/SD19/by_class.zip (the `by_class` folder should be at root level)

Run experiments:

```
python benchmark.py
```

Files explanation:

- `dataset.py` - preprocess and load dataset, prepare it for experiments
- `hog.py` - implementation of the histogram of oriented gradients method
- `cnn.py` - implementation of simple convolutional neural network
- `benchmark.py` - run training/inference and compare the different methods on a test subset of the dataset