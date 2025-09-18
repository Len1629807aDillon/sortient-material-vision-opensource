"""Sensor calibration workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from ..utils.logging import get_logger
from .sensor_models import SensorModel

logger = get_logger(__name__)


@dataclass(slots=True)
class CalibrationResult:
    sensor_name: str
    bias: float
    gain: float
    residual_error: float


def calibrate_sensor(sensor: SensorModel, reference: np.ndarray, samples: Iterable[np.ndarray]) -> CalibrationResult:
    reference_mean = reference.mean()
    sample_mean = np.mean([sample.mean() for sample in samples])
    bias = reference_mean - sample_mean
    gain = reference.std() / (np.std([sample.std() for sample in samples]) + 1e-8)
    residual = float(np.abs(bias) + np.abs(1 - gain))
    logger.info("Calibrated sensor %s bias=%.4f gain=%.4f", sensor.specification.name, bias, gain)
    return CalibrationResult(sensor.specification.name, float(bias), float(gain), residual)
