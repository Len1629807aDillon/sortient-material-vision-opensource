"""Sensor modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SensorSpecification:
    name: str
    resolution: Tuple[int, int]
    spectral_range: Tuple[float, float]
    noise_level: float
    exposure: float


class SensorModel:
    """Base class representing sensor characteristics."""

    def __init__(self, specification: SensorSpecification) -> None:
        self.specification = specification

    def simulate(self, data: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.specification.noise_level, data.shape)
        simulated = np.clip(data + noise, 0.0, 1.0) * self.specification.exposure
        logger.debug("Simulating %s sensor", self.specification.name)
        return simulated

    def info(self) -> Dict[str, float | str]:
        return {
            "name": self.specification.name,
            "resolution": f"{self.specification.resolution[0]}x{self.specification.resolution[1]}",
            "spectral_start": self.specification.spectral_range[0],
            "spectral_end": self.specification.spectral_range[1],
            "noise_level": self.specification.noise_level,
            "exposure": self.specification.exposure,
        }


class SensorFactory:
    """Factory for building sensor models."""

    @staticmethod
    def create_rgb_sensor() -> SensorModel:
        spec = SensorSpecification("RGB", (2048, 1536), (400, 700), noise_level=0.01, exposure=1.0)
        return SensorModel(spec)

    @staticmethod
    def create_nir_sensor() -> SensorModel:
        spec = SensorSpecification("NIR", (1024, 768), (900, 1700), noise_level=0.02, exposure=1.1)
        return SensorModel(spec)

    @staticmethod
    def create_hyperspectral_sensor(bands: int = 240) -> SensorModel:
        spec = SensorSpecification(
            f"Hyperspectral-{bands}",
            (512, 512),
            (400, 2500),
            noise_level=0.03,
            exposure=0.95,
        )
        return SensorModel(spec)
