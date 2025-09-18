"""Training loop implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..data.augmentations import build_standard_augmentations
from ..data.loader import DataLoader, build_dataloaders
from ..models.multispectral_classifier import MultiSpectralClassifier
from ..utils.config import ExperimentConfig
from ..utils.logging import get_logger
from .callbacks import EarlyStopping, MetricLogger, TrainingCallback
from .schedulers import CosineScheduler, WarmupScheduler

logger = get_logger(__name__)


@dataclass(slots=True)
class Trainer:
    config: ExperimentConfig
    dataloaders: Dict[str, DataLoader]
    model: MultiSpectralClassifier
    callbacks: List[TrainingCallback] = field(default_factory=list)
    scheduler: WarmupScheduler | None = None

    epoch: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    label_to_index: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config_dict: Dict[str, object]) -> "Trainer":
        config = ExperimentConfig().merge(config_dict)
        dataloaders = build_dataloaders(
            config.data,
            augmentation_factory=build_standard_augmentations if config.data.get("augment", True) else None,
        )
        model = MultiSpectralClassifier.from_config(config.model)
        callbacks: List[TrainingCallback] = [MetricLogger()]
        if config.training.get("early_stopping", True):
            callbacks.append(EarlyStopping())
        scheduler = None
        if config.training.get("scheduler", True):
            cosine = CosineScheduler(
                initial_lr=float(config.training.get("lr", 0.05)),
                min_lr=float(config.training.get("min_lr", 0.001)),
                total_epochs=int(config.training.get("epochs", 10)),
            )
            scheduler = WarmupScheduler(
                warmup_epochs=int(config.training.get("warmup_epochs", 2)),
                target_lr=float(config.training.get("lr", 0.05)),
                after=cosine,
            )
        trainer = cls(
            config=config,
            dataloaders=dataloaders,
            model=model,
            callbacks=callbacks,
            scheduler=scheduler,
        )
        trainer._init_label_mapping()
        return trainer

    def _init_label_mapping(self) -> None:
        labels = set()
        for split in self.dataloaders.values():
            for batch in split:
                for sample in batch:
                    labels.add(sample.label)
        if not labels:
            labels = {"unknown"}
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(labels))}
        logger.info("Label mapping: %s", self.label_to_index)

    def fit(self, dry_run: bool = False) -> None:
        epochs = int(self.config.training.get("epochs", 10))
        learning_rate = float(self.config.training.get("lr", 0.05))
        if dry_run:
            logger.info("Dry-run mode: iterating configuration without heavy computation")
        for epoch in range(epochs):
            self.epoch = epoch + 1
            lr = self.scheduler.value(epoch) if self.scheduler else learning_rate
            logger.debug("Epoch %d learning rate %.6f", self.epoch, lr)
            for callback in self.callbacks:
                if hasattr(callback, "on_train_epoch_start"):
                    callback.on_train_epoch_start(self)
            if dry_run:
                self.metrics = {"train_loss": 0.0, "val_loss": 0.0, "val_accuracy": 0.0}
            else:
                train_loss = self._train_epoch(lr)
                val_loss, val_accuracy = self._validate_epoch()
                self.metrics = {
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_accuracy),
                }
            for callback in self.callbacks:
                try:
                    if hasattr(callback, "on_train_epoch_end"):
                        callback.on_train_epoch_end(self)
                except StopIteration:
                    logger.info("Early stopping at epoch %d", self.epoch)
                    return

    # ----------------------------------------------------------------------------------

    def _train_epoch(self, learning_rate: float) -> float:
        if not self.model.uses_numpy_backend():
            logger.warning("Training is currently only implemented for the NumPy backend")
            return 0.0
        weights = self.model.numpy_weights()
        total_loss = 0.0
        total_samples = 0
        for batch in self.dataloaders.get("train", []):
            features, labels, targets_extra = self._prepare_batch(batch)
            if features.size == 0:
                continue
            logits = weights[: self.model.num_classes] @ features.T
            logits = logits.T
            logits -= logits.max(axis=1, keepdims=True)
            exp_scores = np.exp(logits)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            one_hot = np.eye(self.model.num_classes)[labels]
            classification_loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-12), axis=1))
            grad_cls = (probs - one_hot).T @ features / features.shape[0]

            preds_extra = weights[self.model.num_classes :] @ features.T
            preds_extra = preds_extra.T
            extra_loss = np.mean((preds_extra - targets_extra) ** 2)
            grad_extra = 2 * (preds_extra - targets_extra).T @ features / features.shape[0]

            weights[: self.model.num_classes] -= learning_rate * grad_cls
            weights[self.model.num_classes :] -= learning_rate * grad_extra

            total_loss += float(classification_loss + extra_loss) * features.shape[0]
            total_samples += features.shape[0]
        if total_samples > 0:
            total_loss /= total_samples
        self.model.update_numpy_weights(weights)
        return total_loss

    def _validate_epoch(self) -> tuple[float, float]:
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        for batch in self.dataloaders.get("val", []):
            features, labels, targets_extra = self._prepare_batch(batch)
            if features.size == 0:
                continue
            outputs = []
            for feature in features:
                logits = self.model.numpy_weights() @ feature
                outputs.append(logits)
            outputs = np.asarray(outputs)
            cls_logits = outputs[:, : self.model.num_classes]
            probs = np.exp(cls_logits - cls_logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            one_hot = np.eye(self.model.num_classes)[labels]
            loss_cls = -np.mean(np.sum(one_hot * np.log(probs + 1e-12), axis=1))
            preds_extra = outputs[:, self.model.num_classes :]
            loss_extra = np.mean((preds_extra - targets_extra) ** 2)
            total_loss += float(loss_cls + loss_extra) * features.shape[0]
            predictions = probs.argmax(axis=1)
            total_accuracy += float(np.mean(predictions == labels)) * features.shape[0]
            total_samples += features.shape[0]
        if total_samples == 0:
            return 0.0, 0.0
        return total_loss / total_samples, total_accuracy / total_samples

    def _prepare_batch(self, batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature_list: List[np.ndarray] = []
        label_list: List[int] = []
        extra_targets: List[np.ndarray] = []
        for sample in batch:
            normalized = sample.tensor.normalized()
            feature = self.model.backbone.fuse(normalized.rgb, normalized.nir, normalized.hyperspectral)
            feature_list.append(feature)
            label_list.append(self.label_to_index.get(sample.label, 0))
            extra_targets.append(
                np.array([
                    float(sample.recyclable),
                    float(sample.contamination_score),
                    float(sample.quality_score),
                ])
            )
        if not feature_list:
            return np.empty((0, 0)), np.empty((0,), dtype=int), np.empty((0, 3))
        features = np.stack(feature_list, axis=0)
        labels = np.asarray(label_list, dtype=int)
        extras = np.stack(extra_targets, axis=0)
        return features, labels, extras
