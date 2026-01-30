from abc import ABC, abstractmethod
import numpy as np
import time
from tqdm import tqdm


class BaseClassicalOCR(ABC):
    """
    Enhanced base class for classical OCR methods with preprocessing support.

    Extends the BaseOCR pattern with preprocessing capabilities for
    classical computer vision methods.
    """

    def __init__(self, preprocessing_config=None):
        """
        Args:
            preprocessing_config: Dict with preprocessing options or None to skip preprocessing
        """
        self.preprocessing_config = preprocessing_config
        self.preprocessor = None

        # Create preprocessor if config provided
        if preprocessing_config:
            from preprocessing.image_preprocessing import ImagePreprocessor
            self.preprocessor = ImagePreprocessor(preprocessing_config)

        self.classifier = None

    def preprocess_images(self, images):
        """
        Apply preprocessing pipeline to images.

        Args:
            images: List of images (numpy arrays or tensors)

        Returns:
            List of preprocessed images
        """
        if self.preprocessor is None:
            return images

        return [self.preprocessor.process(img)
                for img in tqdm(images, desc="Preprocessing images")]

    @abstractmethod
    def extract_features(self, images):
        """
        Extract features from images. Must be implemented by subclasses.

        Args:
            images: List of preprocessed images

        Returns:
            numpy array of shape (n_images, n_features)
        """
        raise NotImplementedError

    def fit(self, X_images, y):
        """
        Train the classifier on preprocessed images.

        Args:
            X_images: List of raw images
            y: Labels
        """
        # 1. Preprocess
        print(f"[{self.__class__.__name__}] Preprocessing {len(X_images)} training images...")
        X_preprocessed = self.preprocess_images(X_images)

        # 2. Extract features
        X_features = self.extract_features(X_preprocessed)

        # 3. Train classifier
        print(f"[{self.__class__.__name__}] Training classifier...")
        start_time = time.time()
        self.classifier.fit(X_features, y)
        print(f"Training finished in {time.time() - start_time:.2f}s")

    def predict(self, X_images, return_confidence=False):
        """
        Predict labels for new images.

        Args:
            X_images: List of raw images
            return_confidence: If True, also return confidence scores

        Returns:
            predictions (and optionally confidences)
        """
        # Preprocess
        X_preprocessed = self.preprocess_images(X_images)

        # Extract features
        X_features = self.extract_features(X_preprocessed)

        # Predict
        predictions = self.classifier.predict(X_features)

        if return_confidence:
            if hasattr(self.classifier, 'predict_proba'):
                probs = self.classifier.predict_proba(X_features)
                confidences = np.max(probs, axis=1)
            elif hasattr(self.classifier, 'decision_function'):
                # For LinearSVC and other classifiers without predict_proba
                decision = self.classifier.decision_function(X_features)
                # Convert decision function to pseudo-confidence (normalized)
                if decision.ndim == 1:
                    # Binary classification
                    confidences = np.abs(decision)
                else:
                    # Multi-class: use max decision value
                    confidences = np.max(decision, axis=1)
                # Normalize to [0, 1] range
                confidences = (confidences - confidences.min()) / (confidences.max() - confidences.min() + 1e-8)
            else:
                # No confidence available, return ones
                confidences = np.ones(len(predictions))
            return predictions, confidences

        return predictions
