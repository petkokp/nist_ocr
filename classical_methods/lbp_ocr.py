import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from tqdm import tqdm

from .base_classical import BaseClassicalOCR


class LBPSVM_OCR(BaseClassicalOCR):
    """
    OCR using Local Binary Patterns (LBP) + Linear SVM.

    LBP captures local texture patterns by binarizing a neighborhood of pixels.
    """

    def __init__(self, radius=2, n_points=None, method="uniform",
                 class_weight="balanced", max_iter=2000, preprocessing_config=None):
        """
        Args:
            radius: Radius for LBP sampling
            n_points: Number of points for LBP
            method: LBP method
            class_weight: 'balanced' to handle class imbalance, or None
            max_iter: Max number of iterations 
            preprocessing_config: preprocessing config
        """
        super().__init__(preprocessing_config)

        self.radius = radius
        self.n_points = n_points if n_points is not None else radius * 8 # idk, can change default
        self.method = method

        self.classifier = LinearSVC(dual="auto", max_iter=max_iter, class_weight=class_weight)

    def extract_features(self, images):
        """
        Extract LBP histogram features from images.

        Returns:
            numpy array of shape (n_images, n_features)
        """
        features = []

        for img in tqdm(images, desc="Extracting LBP features"):
            if hasattr(img, "numpy"):
                img = img.numpy().squeeze()

            if img.ndim == 3:
                img = img.squeeze()

            if img.max() > 1.0:
                img = img / 255.0

            lbp = local_binary_pattern(img, P=self.n_points, R=self.radius, method=self.method)

            if self.method == "uniform":
                n_bins = self.n_points + 2
            else:
                n_bins = int(lbp.max() + 1)

            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=True)
            features.append(hist.astype(np.float32))

        feature_array = np.array(features)
        print(f"LBP feature extraction complete. Vector size: {feature_array.shape[1]}")
        return feature_array
