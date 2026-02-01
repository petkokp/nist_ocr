import numpy as np
import mahotas
from sklearn.svm import SVC
from tqdm.contrib.concurrent import process_map
from functools import partial
from .base_classical import BaseClassicalOCR

def _compute_single_zernike(img, radius, degree):
    if hasattr(img, 'numpy'):
        img = img.numpy().squeeze()
    
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    try:
        return mahotas.features.zernike_moments(img, radius=radius, degree=degree)
    except Exception:
        n_features = (degree + 1) * (degree + 2) // 2
        return np.zeros(n_features)

class ZernikeOCR(BaseClassicalOCR):
    def __init__(self, radius=30, degree=12, preprocessing_config=None,
                 kernel='rbf', C=10.0, gamma='scale', class_weight='balanced'):
        super().__init__(preprocessing_config)
        self.radius = radius
        self.degree = degree
        self.classifier = SVC(kernel=kernel, C=C, gamma=gamma,
                             probability=True, random_state=42,
                             class_weight=class_weight, verbose=True)

    def extract_features(self, images):
        worker = partial(_compute_single_zernike, radius=self.radius, degree=self.degree)

        features = process_map(worker, images, chunksize=10, 
                               desc="Extracting Zernike moments (Parallel)")

        feature_array = np.array(features)
        print(f"Feature extraction complete. Shape: {feature_array.shape}")
        return feature_array