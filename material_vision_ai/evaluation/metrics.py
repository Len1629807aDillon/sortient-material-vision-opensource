"""Evaluation metrics used across the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Any

import numpy as np


@dataclass(slots=True)
class MetricResult:
    name: str
    value: float
    details: Dict[str, Any]


def accuracy(predictions: Iterable[int], targets: Iterable[int]) -> MetricResult:
    preds = np.asarray(list(predictions))
    labels = np.asarray(list(targets))
    if preds.size == 0:
        return MetricResult("accuracy", 0.0, {"count": 0})
    value = float(np.mean(preds == labels))
    return MetricResult("accuracy", value, {"count": float(preds.size)})


def confusion_matrix(predictions: Iterable[int], targets: Iterable[int], num_classes: int) -> MetricResult:
    preds = np.asarray(list(predictions))
    labels = np.asarray(list(targets))
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(preds, labels):
        matrix[label, pred] += 1
    return MetricResult("confusion_matrix", float(np.trace(matrix)), {"matrix": matrix})


def mean_absolute_error(predictions: Iterable[float], targets: Iterable[float]) -> MetricResult:
    preds = np.asarray(list(predictions))
    labels = np.asarray(list(targets))
    if preds.size == 0:
        return MetricResult("mae", 0.0, {})
    value = float(np.mean(np.abs(preds - labels)))
    return MetricResult("mae", value, {"count": float(preds.size)})
