"""Inference pipeline for Material Vision AI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from ..data.dataset import MaterialDataset, SpectralTensor, MaterialSample
from ..models.multispectral_classifier import MultiSpectralClassifier
from ..utils.config import ExperimentConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class InferenceResult:
    label: str
    confidence: float
    recyclability: float
    contamination_score: float
    quality_score: float


class InferencePipeline:
    """High-level inference pipeline."""

    def __init__(self, model: MultiSpectralClassifier, class_names: List[str]) -> None:
        self.model = model
        self.class_names = class_names

    @classmethod
    def from_config(cls, config_dict: Dict[str, object]) -> "InferencePipeline":
        config = ExperimentConfig().merge(config_dict)
        model = MultiSpectralClassifier.from_config(config.model)
        class_names = list(config.model.get("class_names", [f"class_{i}" for i in range(model.num_classes)]))
        return cls(model=model, class_names=class_names)

    def run(self, manifest: Path | None, dry_run: bool = False) -> Dict[str, List[Dict[str, float | str]]]:
        if dry_run:
            logger.info("Dry-run inference: returning configuration metadata only")
            return {"predictions": []}
        samples = self._load_samples(manifest)
        return {"predictions": self.predict_samples(samples)}

    def predict_samples(self, samples: List[MaterialSample]) -> List[Dict[str, float | str]]:
        predictions: List[Dict[str, float | str]] = []
        for sample in samples:
            normalized = sample.tensor.normalized()
            output = self.model.forward(normalized.rgb, normalized.nir, normalized.hyperspectral)
            probs = self._softmax(output.material_logits)
            label_index = int(probs.argmax())
            predictions.append(
                {
                    "label": self.class_names[label_index] if label_index < len(self.class_names) else str(label_index),
                    "confidence": float(probs[label_index]),
                    "recyclability": float(output.recyclability),
                    "contamination_score": float(output.contamination_score),
                    "quality_score": float(output.quality_score),
                }
            )
        return predictions

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def _load_samples(self, manifest: Path | None) -> List[MaterialSample]:
        if manifest is None:
            raise ValueError("Inference requires an input manifest path")
        if manifest.is_dir():
            dataset = MaterialDataset.from_directory(manifest)
            return list(dataset)
        data = json.loads(manifest.read_text())
        samples: List[MaterialSample] = []
        for entry in data.get("samples", []):
            tensor = SpectralTensor(
                rgb=np.asarray(entry["rgb"], dtype=float),
                nir=np.asarray(entry["nir"], dtype=float),
                hyperspectral=np.asarray(entry["hyperspectral"], dtype=float),
                metadata=entry.get("metadata", {}),
            )
            samples.append(
                MaterialSample(
                    tensor=tensor,
                    label=str(entry.get("label", "unknown")),
                    recyclable=bool(entry.get("recyclable", True)),
                    quality_score=float(entry.get("quality_score", 0.0)),
                    contamination_score=float(entry.get("contamination_score", 0.0)),
                )
            )
        return samples
