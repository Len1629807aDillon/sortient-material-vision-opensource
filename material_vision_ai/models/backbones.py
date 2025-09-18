"""Model backbones for spectral fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - fallback to numpy implementation
    torch = None
    nn = object  # type: ignore
    F = None


@dataclass(slots=True)
class BackboneRegistryEntry:
    name: str
    builder: Callable[..., "BaseBackbone"]
    description: str


class BackboneRegistry:
    """Registry for backbone architectures."""

    def __init__(self) -> None:
        self._registry: Dict[str, BackboneRegistryEntry] = {}

    def register(self, name: str, builder: Callable[..., "BaseBackbone"], description: str) -> None:
        if name in self._registry:
            raise ValueError(f"Backbone {name} already registered")
        self._registry[name] = BackboneRegistryEntry(name, builder, description)
        logger.info("Registered backbone %s", name)

    def create(self, name: str, **kwargs) -> "BaseBackbone":
        if name not in self._registry:
            raise KeyError(f"Unknown backbone {name}")
        return self._registry[name].builder(**kwargs)

    def info(self) -> Dict[str, str]:
        return {name: entry.description for name, entry in self._registry.items()}


BACKBONES = BackboneRegistry()


class BaseBackbone:
    """Base class for feature extractors."""

    def fuse(self, rgb: np.ndarray, nir: np.ndarray, hyperspectral: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def feature_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError


class TorchSpectralBackbone(BaseBackbone, nn.Module if torch else object):  # type: ignore[misc]
    """A PyTorch-based spectral fusion backbone."""

    def __init__(self, channels: int = 64) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for TorchSpectralBackbone")
        super().__init__()
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.nir_conv = nn.Sequential(
            nn.Conv2d(1, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )
        self.hs_conv = nn.Sequential(
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3 - channels // 2, channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self._out_channels = channels * 2

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor, hyperspectral: torch.Tensor) -> torch.Tensor:
        hs = hyperspectral
        if hs.shape[1] != 32:
            hs = F.interpolate(hs, size=(32, rgb.shape[2], rgb.shape[3]))
        rgb_feat = self.rgb_conv(rgb)
        nir_feat = self.nir_conv(nir)
        hs_feat = self.hs_conv(hs)
        fused = torch.cat([rgb_feat, nir_feat, hs_feat], dim=1)
        return self.fusion(fused)

    # BaseBackbone compliance ---------------------------------------------------------------

    def fuse(self, rgb: np.ndarray, nir: np.ndarray, hyperspectral: np.ndarray) -> np.ndarray:  # type: ignore[override]
        rgb_t = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float()
        nir_t = torch.tensor(nir).unsqueeze(0).unsqueeze(0).float()
        hs_t = torch.tensor(hyperspectral).unsqueeze(0).float()
        with torch.no_grad():
            fused = self.forward(rgb_t, nir_t, hs_t)
            pooled = torch.mean(fused, dim=(2, 3))
        return pooled.squeeze(0).cpu().numpy()

    def feature_shape(self) -> Tuple[int, ...]:  # type: ignore[override]
        return (self._out_channels,)


class NumpySpectralBackbone(BaseBackbone):
    """Fallback backbone implemented with NumPy operations."""

    def __init__(self, channels: int = 32) -> None:
        self.channels = channels
        self._feature_shape = (channels * 2,)

    def fuse(self, rgb: np.ndarray, nir: np.ndarray, hyperspectral: np.ndarray) -> np.ndarray:
        rgb_mean = rgb.mean(axis=(0, 1))
        nir_stats = np.array([nir.mean(), nir.std()])
        hs_mean = hyperspectral.mean(axis=(1, 2))
        features = np.concatenate([rgb_mean, nir_stats, hs_mean])
        if features.size < self._feature_shape[0]:
            padding = np.zeros(self._feature_shape[0] - features.size)
            features = np.concatenate([features, padding])
        return features[: self._feature_shape[0]]

    def feature_shape(self) -> Tuple[int, ...]:
        return self._feature_shape


def register_default_backbones() -> None:
    """Register built-in backbones."""

    if torch is not None:
        BACKBONES.register(
            "torch_spectral",
            lambda channels=64: TorchSpectralBackbone(channels=channels),
            "PyTorch spectral fusion backbone with learnable convolutions.",
        )
    BACKBONES.register(
        "numpy_spectral",
        lambda channels=32: NumpySpectralBackbone(channels=channels),
        "NumPy-based fallback spectral feature extractor for lightweight environments.",
    )


register_default_backbones()
