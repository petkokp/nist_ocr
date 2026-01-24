import torch
import torchvision.transforms as transforms
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


class AugmentationPipeline:
    """
    Data augmentation pipeline for CNN training.

    Includes:
    - Random rotations
    - Elastic distortions
    - Affine transformations
    - Random noise
    """

    def __init__(self, config):
        """
        Args:
            config: Dict with augmentation parameters:
                - rotation_range: float (degrees, e.g., 15)
                - elastic_alpha: float (distortion strength, e.g., 34)
                - elastic_sigma: float (smoothness, e.g., 4)
                - translation_range: float (fraction, e.g., 0.1)
                - scale_range: tuple (e.g., (0.9, 1.1))
                - noise_std: float (e.g., 0.05)
                - apply_prob: float (probability of applying each transform)
        """
        self.config = config

        # Build torchvision transforms
        self.transforms = self._build_transforms()

    def _build_transforms(self):
        """Build torchvision transform pipeline."""
        transform_list = []

        # Random rotation
        if self.config.get('rotation_range', 0) > 0:
            rotation = self.config['rotation_range']
            transform_list.append(
                transforms.RandomRotation(degrees=(-rotation, rotation))
            )

        # Random affine (translation + scale)
        translate = self.config.get('translation_range', 0.0)
        scale_range = self.config.get('scale_range', (1.0, 1.0))
        if translate > 0 or scale_range != (1.0, 1.0):
            transform_list.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(translate, translate),
                    scale=scale_range
                )
            )

        return transforms.Compose(transform_list) if transform_list else None

    def __call__(self, image):
        """Apply augmentation to a single image."""
        # Apply torchvision transforms
        if self.transforms is not None:
            image = self.transforms(image)

        # Apply elastic distortion with probability
        if (self.config.get('elastic_alpha', 0) > 0 and
            np.random.rand() < self.config.get('apply_prob', 0.5)):
            image = self._elastic_distortion(image)

        # Add Gaussian noise
        if (self.config.get('noise_std', 0) > 0 and
            np.random.rand() < self.config.get('apply_prob', 0.5)):
            image = self._add_noise(image)

        return image

    def _elastic_distortion(self, image):
        """
        Apply elastic distortion to image.

        This is particularly effective for handwritten character recognition
        as it simulates natural variations in writing.
        """
        alpha = self.config.get('elastic_alpha', 34)
        sigma = self.config.get('elastic_sigma', 4)

        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            img_np = image.numpy().squeeze()
            is_tensor = True
        else:
            img_np = np.array(image).squeeze()
            is_tensor = False

        shape = img_np.shape

        # Generate random displacement fields
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        # Create meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

        # Apply distortion
        distorted = map_coordinates(img_np, indices, order=1, mode='reflect')
        distorted = distorted.reshape(shape)

        # Convert back to tensor if needed
        if is_tensor:
            return torch.from_numpy(distorted).unsqueeze(0).float()
        return distorted

    def _add_noise(self, image):
        """Add Gaussian noise to image."""
        noise_std = self.config.get('noise_std', 0.05)

        if isinstance(image, torch.Tensor):
            noise = torch.randn_like(image) * noise_std
            return torch.clamp(image + noise, 0, 1)
        else:
            noise = np.random.randn(*image.shape) * noise_std
            return np.clip(image + noise, 0, 1)
