"""Learning rate scheduling utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class CosineScheduler:
    """Implements cosine annealing for scalar parameters."""

    initial_lr: float
    min_lr: float
    total_epochs: int

    def value(self, epoch: int) -> float:
        if self.total_epochs <= 0:
            return self.initial_lr
        progress = min(max(epoch / self.total_epochs, 0.0), 1.0)
        cosine = (1 + math.cos(math.pi * progress)) / 2
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine


@dataclass(slots=True)
class WarmupScheduler:
    """Warmup scheduler followed by another schedule."""

    warmup_epochs: int
    target_lr: float
    after: CosineScheduler

    def value(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.target_lr * (epoch + 1) / max(self.warmup_epochs, 1)
        return self.after.value(epoch - self.warmup_epochs)
