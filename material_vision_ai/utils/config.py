"""Configuration utilities for Material Vision AI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration container for experiments.

    Attributes:
        name: Name of the experiment or run.
        seed: Optional random seed for reproducibility.
        data: Nested dictionary describing data-related parameters.
        model: Nested dictionary describing model architecture parameters.
        training: Nested dictionary describing training hyperparameters.
        evaluation: Nested dictionary with evaluation metrics configuration.
        inference: Nested dictionary controlling inference behavior.
    """

    name: str = "material-vision-experiment"
    seed: int | None = 42
    data: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    inference: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: Dict[str, Any]) -> "ExperimentConfig":
        """Create a new config by merging ``other`` into ``self``."""

        merged = ExperimentConfig(
            name=other.get("name", self.name),
            seed=other.get("seed", self.seed),
            data={**self.data, **other.get("data", {})},
            model={**self.model, **other.get("model", {})},
            training={**self.training, **other.get("training", {})},
            evaluation={**self.evaluation, **other.get("evaluation", {})},
            inference={**self.inference, **other.get("inference", {})},
        )
        return merged


def load_config_file(path: Path) -> Dict[str, Any]:
    """Load YAML or JSON configuration files."""

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported configuration file type: {path.suffix}")


def dump_config(config: Dict[str, Any], path: Path) -> None:
    """Persist configuration dictionary to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(config))
    elif path.suffix.lower() == ".json":
        path.write_text(json.dumps(config, indent=2))
    else:
        raise ValueError(f"Unsupported configuration file type: {path.suffix}")
