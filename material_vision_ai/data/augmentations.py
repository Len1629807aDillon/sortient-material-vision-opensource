"""Data augmentation utilities for multi-spectral inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np

from .dataset import MaterialSample, SpectralTensor
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Augmentation:
    """Base augmentation object."""

    name: str
    probability: float
    apply_fn: Callable[[SpectralTensor], SpectralTensor]

    def __call__(self, tensor: SpectralTensor) -> SpectralTensor:
        if np.random.rand() <= self.probability:
            logger.debug("Applying augmentation %s", self.name)
            return self.apply_fn(tensor)
        return tensor


class AugmentationPipeline:
    """Pipeline applying a sequence of augmentations."""

    def __init__(self, augmentations: Iterable[Augmentation]):
        self._augmentations: List[Augmentation] = list(augmentations)

    def transform(self, sample: MaterialSample) -> MaterialSample:
        tensor = sample.tensor
        for augmentation in self._augmentations:
            tensor = augmentation(tensor)
        return MaterialSample(
            tensor=tensor,
            label=sample.label,
            recyclable=sample.recyclable,
            quality_score=sample.quality_score,
            contamination_score=sample.contamination_score,
        )


# --- Augmentation implementations -----------------------------------------------------------

def random_flip(tensor: SpectralTensor) -> SpectralTensor:
    rgb = np.flip(tensor.rgb, axis=1)
    nir = np.flip(tensor.nir, axis=1)
    hs = np.flip(tensor.hyperspectral, axis=2)
    return SpectralTensor(rgb=rgb, nir=nir, hyperspectral=hs, metadata=tensor.metadata)


def random_rotation(tensor: SpectralTensor, max_angle: float = 15.0) -> SpectralTensor:
    angle = np.random.uniform(-max_angle, max_angle)
    radians = math.radians(angle)
    cos_v, sin_v = np.cos(radians), np.sin(radians)

    def rotate_plane(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        grid_y, grid_x = np.indices((h, w))
        x_shifted = grid_x - cx
        y_shifted = grid_y - cy
        x_rot = x_shifted * cos_v - y_shifted * sin_v + cx
        y_rot = x_shifted * sin_v + y_shifted * cos_v + cy
        x_rot = np.clip(x_rot, 0, w - 1).astype(int)
        y_rot = np.clip(y_rot, 0, h - 1).astype(int)
        return image[y_rot, x_rot]

    rgb = rotate_plane(tensor.rgb)
    nir = rotate_plane(tensor.nir)
    hs = np.stack([rotate_plane(band) for band in tensor.hyperspectral], axis=0)
    return SpectralTensor(rgb=rgb, nir=nir, hyperspectral=hs, metadata=tensor.metadata)


def gaussian_noise(tensor: SpectralTensor, sigma: float = 0.01) -> SpectralTensor:
    rgb = np.clip(tensor.rgb + np.random.normal(0, sigma, tensor.rgb.shape), 0.0, 1.0)
    nir = np.clip(tensor.nir + np.random.normal(0, sigma, tensor.nir.shape), 0.0, 1.0)
    hs = np.clip(tensor.hyperspectral + np.random.normal(0, sigma, tensor.hyperspectral.shape), 0.0, 1.0)
    return SpectralTensor(rgb=rgb, nir=nir, hyperspectral=hs, metadata=tensor.metadata)


def build_standard_augmentations() -> AugmentationPipeline:
    """Return a canonical augmentation pipeline used in research papers."""

    augmentations = [
        Augmentation("flip", 0.5, random_flip),
        Augmentation("rotation", 0.5, random_rotation),
        Augmentation("gaussian_noise", 0.3, gaussian_noise),
    ]
    return AugmentationPipeline(augmentations)
