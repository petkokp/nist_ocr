import numpy as np
from sklearn.svm import LinearSVC
from skimage.feature import hog
from tqdm import tqdm

from .base_classical import BaseClassicalOCR


class HOGSVM_OCR(BaseClassicalOCR):
    """
    OCR using HOG (Histogram of Oriented Gradients) features + Linear SVM.

    HOG captures edge and gradient information, effective for character recognition.
    """

    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                 class_weight='balanced', preprocessing_config=None):
        """
        Args:
            orientations: Number of orientation bins for HOG
            pixels_per_cell: Size of cell in pixels
            cells_per_block: Number of cells per block
            class_weight: 'balanced' to handle class imbalance, or None
            preprocessing_config: Optional preprocessing configuration
        """
        super().__init__(preprocessing_config)

        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

        # LinearSVC is faster than SVC for large datasets
        self.classifier = LinearSVC(dual="auto", max_iter=1000, class_weight=class_weight)

    def extract_features(self, images):
        """
        Extract HOG features from images.

        Returns:
            numpy array of shape (n_images, n_features)
        """
        features = []

        for img in tqdm(images, desc="Extracting HOG features"):
            # Ensure image is a numpy array (H x W)
            if hasattr(img, 'numpy'):
                img = img.numpy().squeeze()

            fd = hog(img,
                     orientations=self.orientations,
                     pixels_per_cell=self.pixels_per_cell,
                     cells_per_block=self.cells_per_block,
                     visualize=False)
            features.append(fd)

        feature_array = np.array(features)
        print(f"HOG feature extraction complete. Vector size: {feature_array.shape[1]}")
        return feature_array
