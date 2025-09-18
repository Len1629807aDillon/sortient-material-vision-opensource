"""Data structures for multi-spectrum material datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SpectralTensor:
    """Container for multi-spectral sensor data.

    Attributes:
        rgb: RGB image data as ``(H, W, 3)`` array.
        nir: Near-infrared single-channel array ``(H, W)``.
        hyperspectral: Hyperspectral cube ``(Bands, H, W)``.
        metadata: Arbitrary metadata dictionary.
    """

    rgb: np.ndarray
    nir: np.ndarray
    hyperspectral: np.ndarray
    metadata: Dict[str, float | str | int | bool]

    def validate(self) -> None:
        """Validate shape compatibility and value ranges."""

        logger.debug("Validating spectral tensor metadata=%s", self.metadata)
        if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError("RGB tensor must be HxWx3")
        if self.nir.shape[:2] != self.rgb.shape[:2]:
            raise ValueError("NIR tensor spatial shape must match RGB")
        if self.hyperspectral.ndim != 3:
            raise ValueError("Hyperspectral tensor must be BxHxW")
        if self.hyperspectral.shape[1:] != self.rgb.shape[:2]:
            raise ValueError("Hyperspectral spatial shape must match RGB")
        if np.any(np.isnan(self.rgb)) or np.any(np.isnan(self.nir)):
            raise ValueError("Spectral tensors must not contain NaNs")

    def normalized(self) -> "SpectralTensor":
        """Return a normalized copy of the tensor."""

        rgb = np.clip(self.rgb / 255.0, 0.0, 1.0)
        nir = (self.nir - np.min(self.nir)) / (np.max(self.nir) - np.min(self.nir) + 1e-8)
        hs = self.hyperspectral.astype(np.float32)
        hs -= hs.min(axis=(1, 2), keepdims=True)
        hs /= hs.max(axis=(1, 2), keepdims=True) + 1e-8
        return SpectralTensor(rgb=rgb, nir=nir, hyperspectral=hs, metadata=self.metadata)


@dataclass(slots=True)
class MaterialSample:
    """Represents a labeled material sample."""

    tensor: SpectralTensor
    label: str
    recyclable: bool
    quality_score: float
    contamination_score: float

    def features(self) -> Dict[str, float]:
        """Return derived features used for analytics."""

        return {
            "quality_score": float(self.quality_score),
            "contamination_score": float(self.contamination_score),
            "is_recyclable": float(self.recyclable),
        }


class MaterialDataset:
    """Collection wrapper with convenience utilities."""

    def __init__(self, samples: Iterable[MaterialSample]):
        self._samples: List[MaterialSample] = list(samples)
        logger.debug("Initialized MaterialDataset with %d samples", len(self._samples))

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[MaterialSample]:
        return iter(self._samples)

    def at(self, index: int) -> MaterialSample:
        """Return the sample at ``index``."""

        return self._samples[index]

    def filter(self, *, recyclable: Optional[bool] = None, label: Optional[str] = None) -> "MaterialDataset":
        """Return a filtered dataset."""

        filtered = [
            sample
            for sample in self._samples
            if (recyclable is None or sample.recyclable == recyclable)
            and (label is None or sample.label == label)
        ]
        logger.debug(
            "Filter applied recyclable=%s label=%s resulting=%d",
            recyclable,
            label,
            len(filtered),
        )
        return MaterialDataset(filtered)

    @classmethod
    def from_directory(cls, path: Path) -> "MaterialDataset":
        """Load dataset from directory of ``.npz`` spectral tensors."""

        samples: List[MaterialSample] = []
        if not path.exists():
            logger.warning("Dataset directory %s does not exist; returning empty dataset", path)
            return cls(samples)
        for file in sorted(path.glob("*.npz")):
            with np.load(file) as data:
                metadata_raw = data["metadata"] if "metadata" in data else b"{}"
                tensor = SpectralTensor(
                    rgb=data["rgb"],
                    nir=data["nir"],
                    hyperspectral=data["hyperspectral"],
                    metadata=json_load_safe(metadata_raw),
                )
                tensor.validate()
                samples.append(
                    MaterialSample(
                        tensor=tensor,
                        label=str(data["label"]),
                        recyclable=bool(data["recyclable"]),
                        quality_score=float(data["quality_score"]),
                        contamination_score=float(data["contamination_score"]),
                    )
                )
        logger.info("Loaded %d material samples from %s", len(samples), path)
        return cls(samples)


def json_load_safe(value: bytes | str | np.ndarray) -> Dict[str, str | float | int | bool]:
    """Best-effort JSON decoding supporting ``numpy``-stored metadata."""

    if isinstance(value, np.ndarray):
        value = value.item() if value.size == 1 else value.tobytes()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Failed to parse metadata JSON: %s", exc)
        return {}
