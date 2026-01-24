import numpy as np
import cv2
from skimage import filters, morphology


class ImagePreprocessor:
    """
    Comprehensive preprocessing pipeline for classical OCR methods.

    Includes adaptive thresholding, morphological operations,
    deskewing, and normalization.
    """

    def __init__(self, config):
        """
        Args:
            config: Dictionary with preprocessing options:
                - adaptive_threshold: bool
                - threshold_method: 'otsu' or 'sauvola'
                - morphology: bool
                - morph_operation: 'opening', 'closing', or 'both'
                - morph_kernel_size: int (default 3)
                - normalize: bool
                - deskew: bool
        """
        self.config = config

    def process(self, image):
        """Apply full preprocessing pipeline."""
        img = self._ensure_numpy(image)

        # 1. Adaptive thresholding
        if self.config.get('adaptive_threshold', False):
            img = self._adaptive_threshold(img)

        # 2. Morphological operations
        if self.config.get('morphology', False):
            img = self._morphological_operations(img)

        # 3. Deskew
        if self.config.get('deskew', False):
            img = self._deskew(img)

        # 4. Normalize
        if self.config.get('normalize', True):
            img = self._normalize(img)

        return img

    def _ensure_numpy(self, image):
        """Convert image to numpy array."""
        if hasattr(image, 'numpy'):
            return image.numpy().squeeze()
        return np.array(image).squeeze()

    def _adaptive_threshold(self, image):
        """Apply adaptive thresholding."""
        method = self.config.get('threshold_method', 'otsu')

        # Ensure image is in [0, 255] range for thresholding
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        if method == 'otsu':
            # Otsu's method: automatic threshold selection
            thresh = filters.threshold_otsu(image)
            binary = image > thresh
        elif method == 'sauvola':
            # Sauvola's method: local adaptive thresholding
            window_size = self.config.get('sauvola_window_size', 15)
            thresh = filters.threshold_sauvola(image, window_size=window_size)
            binary = image > thresh
        else:
            return image

        return binary.astype(np.float32)

    def _morphological_operations(self, image):
        """Apply morphological operations to clean up the image."""
        operation = self.config.get('morph_operation', 'opening')
        kernel_size = self.config.get('morph_kernel_size', 3)

        # Convert to binary if needed
        if image.max() <= 1.0 and image.min() >= 0.0:
            binary = (image > 0.5).astype(np.uint8)
        else:
            binary = (image > 128).astype(np.uint8)

        # Create structuring element
        selem = morphology.disk(kernel_size // 2)

        if operation == 'opening':
            # Opening: erosion followed by dilation (removes small noise)
            result = morphology.opening(binary, selem)
        elif operation == 'closing':
            # Closing: dilation followed by erosion (fills small holes)
            result = morphology.closing(binary, selem)
        elif operation == 'both':
            # Apply both operations
            result = morphology.opening(binary, selem)
            result = morphology.closing(result, selem)
        else:
            result = binary

        return result.astype(np.float32)

    def _deskew(self, image):
        """
        Deskew image using moments.

        Computes the skew angle from image moments and rotates to correct.
        """
        # Ensure binary image
        if image.max() <= 1.0:
            binary = (image > 0.5).astype(np.uint8) * 255
        else:
            binary = (image > 128).astype(np.uint8) * 255

        # Calculate moments
        moments = cv2.moments(binary)

        # Avoid division by zero
        if abs(moments['mu02']) < 1e-2:
            return image

        # Calculate skew angle
        skew = moments['mu11'] / moments['mu02']
        angle = 0.5 * np.arctan(2 * skew) * 180 / np.pi

        # Only deskew if angle is significant (> 1 degree)
        if abs(angle) > 1.0:
            # Rotate to correct skew
            h, w = image.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

        return image

    def _normalize(self, image):
        """Normalize image to [0, 1] range."""
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-8:
            return (image - img_min) / (img_max - img_min)
        return image
