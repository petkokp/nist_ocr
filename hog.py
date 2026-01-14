import numpy as np
import time
from sklearn.svm import LinearSVC
from skimage.feature import hog

class BaseOCR:
    def extract_features(self, images):
        raise NotImplementedError("Subclasses must implement feature extraction.")

    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement training.")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement prediction.")

class HOGSVM_OCR(BaseOCR):
    """
    OCR using HOG (Histogram of Oriented Gradients) features + Linear SVM.
    """
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        # HOG Hyperparameters
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        
        # The Classifier (LinearSVC is faster than SVC for large datasets)
        self.classifier = LinearSVC(dual="auto", max_iter=1000)

    def extract_features(self, images):
        """
        Convert a list of raw images (numpy arrays or Tensors) into HOG feature vectors.
        """
        print(f"Extracting HOG features for {len(images)} images...")
        features = []
        
        for i, img in enumerate(images):
            # Ensure image is a numpy array (H x W)
            if hasattr(img, 'numpy'): 
                img = img.numpy().squeeze() # Remove channel dim if Tensor
            
            # Compute HOG
            # visualize=False returns just the vector
            fd = hog(img, 
                     orientations=self.orientations, 
                     pixels_per_cell=self.pixels_per_cell, 
                     cells_per_block=self.cells_per_block, 
                     visualize=False)
            features.append(fd)
            
            # Simple progress tracker
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(images)}", end='\r')
                
        print(f"\nFeature extraction complete. Vector size: {features[0].shape[0]}")
        return np.array(features)

    def fit(self, X_images, y):
        # 1. Extract Features from raw images
        X_features = self.extract_features(X_images)
        
        # 2. Train the SVM
        print("Training SVM classifier...")
        start_time = time.time()
        self.classifier.fit(X_features, y)
        print(f"Training finished in {time.time() - start_time:.2f}s")

    def predict(self, X_images):
        X_features = self.extract_features(X_images)
        return self.classifier.predict(X_features)