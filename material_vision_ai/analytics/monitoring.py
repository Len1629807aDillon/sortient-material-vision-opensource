"""Analytics and monitoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class StreamStatistic:
    name: str
    value: float
    window: int


class PerformanceMonitor:
    """Real-time analytics aggregator for streaming inference."""

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self.records: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            values = self.records.setdefault(key, [])
            values.append(float(value))
            if len(values) > self.window:
                values.pop(0)
        logger.debug("Updated monitor metrics=%s", metrics)

    def summary(self) -> Dict[str, StreamStatistic]:
        summary: Dict[str, StreamStatistic] = {}
        for key, values in self.records.items():
            if not values:
                continue
            summary[key] = StreamStatistic(name=key, value=float(np.mean(values)), window=len(values))
        return summary
