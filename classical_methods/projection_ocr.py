import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from .base_classical import BaseClassicalOCR


class ProjectionOCR(BaseClassicalOCR):
    """
    OCR using Projection Histograms + k-Nearest Neighbors.

    Projection histograms capture the distribution of pixels along
    horizontal and vertical axes, providing simple but effective features.
    """

    def __init__(self, n_neighbors=5, weights='distance', metric='euclidean',
                 preprocessing_config=None, normalize=True):
        """
        Args:
            n_neighbors: Number of neighbors for kNN
            weights: 'uniform' or 'distance'
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            preprocessing_config: Preprocessing options
            normalize: Whether to normalize projection histograms
        """
        super().__init__(preprocessing_config)
        self.n_neighbors = n_neighbors
        self.normalize_proj = normalize

        self.classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=-1  # Use all CPU cores
        )

    def extract_features(self, images):
        """
        Extract horizontal and vertical projection histograms.

        For each image:
        - Horizontal projection: sum of pixels in each row
        - Vertical projection: sum of pixels in each column
        - Concatenate both into single feature vector

        Returns:
            numpy array of shape (n_images, height + width)
        """
        features = []

        for img in tqdm(images, desc="Extracting projection histograms"):
            # Ensure numpy array
            if hasattr(img, 'numpy'):
                img = img.numpy().squeeze()

            # Normalize to [0, 1] if needed
            if img.max() > 1.0:
                img = img / 255.0

            # Compute projections
            h_proj = np.sum(img, axis=1)  # Sum along columns (horizontal projection)
            v_proj = np.sum(img, axis=0)  # Sum along rows (vertical projection)

            # Normalize projections if requested
            if self.normalize_proj:
                h_proj = h_proj / (np.sum(h_proj) + 1e-8)
                v_proj = v_proj / (np.sum(v_proj) + 1e-8)

            # Concatenate
            feature_vector = np.concatenate([h_proj, v_proj])
            features.append(feature_vector)

        feature_array = np.array(features)
        print(f"Projection histogram extraction complete. Vector size: {feature_array.shape[1]}")
        return feature_array
