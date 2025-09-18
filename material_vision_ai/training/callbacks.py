"""Training callbacks for Material Vision AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol

from ..utils.logging import get_logger

logger = get_logger(__name__)


class TrainerProtocol(Protocol):
    epoch: int
    metrics: Dict[str, float]


class TrainingCallback(Protocol):
    """Interface for training callbacks."""

    def on_train_epoch_start(self, trainer: TrainerProtocol) -> None:  # pragma: no cover - default empty
        ...

    def on_train_epoch_end(self, trainer: TrainerProtocol) -> None:  # pragma: no cover - default empty
        ...


@dataclass(slots=True)
class MetricLogger:
    """Logs metrics after each epoch."""

    history: Dict[str, list] = field(default_factory=lambda: {})

    def on_train_epoch_end(self, trainer: TrainerProtocol) -> None:  # pragma: no cover - simple logging
        for key, value in trainer.metrics.items():
            self.history.setdefault(key, []).append(value)
            logger.info("Epoch %d - %s: %.4f", trainer.epoch, key, value)


@dataclass(slots=True)
class EarlyStopping:
    """Early stopping based on validation metric."""

    monitor: str = "val_loss"
    patience: int = 5
    mode: str = "min"
    best_score: float | None = None
    wait: int = 0
    stopped_epoch: int = 0

    def on_train_epoch_end(self, trainer: TrainerProtocol) -> None:
        metric = trainer.metrics.get(self.monitor)
        if metric is None:
            logger.debug("Metric %s not found; skipping early stopping", self.monitor)
            return
        if self.best_score is None:
            self.best_score = metric
            logger.debug("Setting initial best score %.4f", metric)
            return
        improvement = (metric < self.best_score) if self.mode == "min" else (metric > self.best_score)
        if improvement:
            logger.debug("Metric improved from %.4f to %.4f", self.best_score, metric)
            self.best_score = metric
            self.wait = 0
        else:
            self.wait += 1
            logger.debug("No improvement (wait=%d)", self.wait)
            if self.wait >= self.patience:
                self.stopped_epoch = trainer.epoch
                raise StopIteration("Early stopping triggered")
