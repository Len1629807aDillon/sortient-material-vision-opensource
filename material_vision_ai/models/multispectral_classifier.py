"""High-level classifier architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .backbones import BACKBONES, BaseBackbone, TorchSpectralBackbone, torch
from ..utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover
    import torch.nn as nn
except Exception:  # pragma: no cover
    nn = None


@dataclass(slots=True)
class ClassifierOutput:
    material_logits: np.ndarray
    recyclability: float
    contamination_score: float
    quality_score: float


class MultiSpectralClassifier:
    """Classifier supporting both PyTorch and NumPy backends."""

    def __init__(self, backbone: BaseBackbone, num_classes: int) -> None:
        self.backbone = backbone
        self.num_classes = num_classes
        if torch is not None and isinstance(backbone, TorchSpectralBackbone):
            self._head = nn.Sequential(  # type: ignore[arg-type]
                nn.Linear(backbone.feature_shape()[0], num_classes + 3),
            )
            logger.debug("Initialized Torch classifier head for %d classes", num_classes)
        else:
            self._weights = np.random.RandomState(42).normal(size=(num_classes + 3, backbone.feature_shape()[0]))
            logger.debug("Initialized NumPy classifier head for %d classes", num_classes)

    def forward(self, rgb: np.ndarray, nir: np.ndarray, hyperspectral: np.ndarray) -> ClassifierOutput:
        logger.debug("Running forward pass with shapes rgb=%s nir=%s hyperspectral=%s", rgb.shape, nir.shape, hyperspectral.shape)
        features = self.backbone.fuse(rgb, nir, hyperspectral)
        if torch is not None and isinstance(self.backbone, TorchSpectralBackbone):
            logits = self._head(torch.tensor(features).unsqueeze(0)).squeeze(0).detach().cpu().numpy()  # type: ignore[union-attr]
        else:
            logits = self._weights @ features
        material_logits = logits[: self.num_classes]
        extra = logits[self.num_classes : self.num_classes + 3]
        return ClassifierOutput(
            material_logits=material_logits,
            recyclability=float(extra[0]),
            contamination_score=float(extra[1]),
            quality_score=float(extra[2]),
        )

    def uses_numpy_backend(self) -> bool:
        return not (torch is not None and isinstance(self.backbone, TorchSpectralBackbone))

    def numpy_weights(self) -> np.ndarray:
        if not self.uses_numpy_backend():
            raise RuntimeError("NumPy weights are only available for the NumPy backend")
        return self._weights

    def update_numpy_weights(self, new_weights: np.ndarray) -> None:
        if not self.uses_numpy_backend():
            raise RuntimeError("Cannot update NumPy weights when using the Torch backend")
        if new_weights.shape != self._weights.shape:
            raise ValueError("Shape mismatch when updating weights")
        self._weights = new_weights

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "MultiSpectralClassifier":
        backbone_name = str(config.get("backbone", "numpy_spectral"))
        num_classes = int(config.get("num_classes", 10))
        backbone_params = config.get("backbone_params", {})
        backbone = BACKBONES.create(backbone_name, **backbone_params)
        return cls(backbone=backbone, num_classes=num_classes)
