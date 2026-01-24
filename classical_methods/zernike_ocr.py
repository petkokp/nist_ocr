import numpy as np
import mahotas
from sklearn.svm import SVC
from tqdm import tqdm
from .base_classical import BaseClassicalOCR


class ZernikeOCR(BaseClassicalOCR):
    """
    OCR using Zernike Moments (rotation-invariant shape descriptors) + SVM.

    Zernike moments are complex polynomials orthogonal over the unit disk,
    providing rotation-invariant features ideal for character recognition.
    """

    def __init__(self, radius=30, degree=12, preprocessing_config=None,
                 kernel='rbf', C=10.0, gamma='scale', class_weight='balanced'):
        """
        Args:
            radius: Radius for Zernike moment computation
            degree: Maximum degree of Zernike polynomials (higher = more features)
            preprocessing_config: Preprocessing options
            kernel: SVM kernel ('rbf', 'linear', 'poly')
            C: SVM regularization parameter
            gamma: Kernel coefficient
            class_weight: 'balanced' to handle class imbalance, or None
        """
        super().__init__(preprocessing_config)
        self.radius = radius
        self.degree = degree

        # Use SVC (not LinearSVC) to get probability estimates
        # class_weight='balanced' handles imbalanced classes
        self.classifier = SVC(kernel=kernel, C=C, gamma=gamma,
                             probability=True, random_state=42,
                             class_weight=class_weight)

    def extract_features(self, images):
        """
        Extract Zernike moments from images.

        Returns:
            numpy array of shape (n_images, n_features)
            where n_features = (degree+1)*(degree+2)/2
        """
        features = []

        for img in tqdm(images, desc="Extracting Zernike moments"):
            # Ensure numpy array
            if hasattr(img, 'numpy'):
                img = img.numpy().squeeze()

            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            # Compute Zernike moments
            try:
                zm = mahotas.features.zernike_moments(img, radius=self.radius,
                                                      degree=self.degree)
                features.append(zm)
            except Exception as e:
                print(f"Warning: Failed to compute Zernike moments: {e}")
                # Fallback: zero vector
                n_features = (self.degree + 1) * (self.degree + 2) // 2
                features.append(np.zeros(n_features))

        feature_array = np.array(features)
        print(f"Zernike feature extraction complete. Vector size: {feature_array.shape[1]}")
        return feature_array
