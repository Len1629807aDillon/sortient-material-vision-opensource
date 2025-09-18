"""Evaluation reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .metrics import MetricResult


@dataclass(slots=True)
class EvaluationReport:
    metrics: List[MetricResult]

    def summary(self) -> Dict[str, float]:
        return {metric.name: metric.value for metric in self.metrics}

    def detailed(self) -> Dict[str, Dict[str, float]]:
        detailed_report: Dict[str, Dict[str, float]] = {}
        for metric in self.metrics:
            details: Dict[str, float] = {}
            for key, value in metric.details.items():
                if isinstance(value, (int, float)):
                    details[key] = float(value)
            detailed_report[metric.name] = details
        return detailed_report
