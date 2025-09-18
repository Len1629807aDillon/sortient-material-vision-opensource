"""Synthetic dataset generation aligned with the Sortient material catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..catalog.database import MATERIAL_DATABASE, MaterialRecord
from .dataset import MaterialDataset, MaterialSample, SpectralTensor


@dataclass(slots=True)
class SyntheticMaterialProfile:
    """Parameterized profile describing synthetic material behavior."""

    identifier: str
    label: str
    noise_scale: float
    contamination_chance: float
    recyclability_bias: float
    quality_baseline: float
    spectral_gain: float
    spectral_jitter: float

    def base_record(self) -> MaterialRecord:
        if self.identifier not in MATERIAL_DATABASE:
            raise KeyError(f"Profile {self.identifier} not present in catalog")
        return MATERIAL_DATABASE[self.identifier]

    def sample_metadata(self, rng: np.random.Generator) -> Dict[str, float | str]:
        record = self.base_record()
        return {
            "catalog_id": self.identifier,
            "category": record["category"],
            "density": float(record["density"]),
            "stability_factor": float(record["stability_factor"]),
            "sustainability_index": float(record["sustainability_index"]),
            "noise_scale": float(self.noise_scale),
            "spectral_gain": float(self.spectral_gain),
            "spectral_jitter": float(self.spectral_jitter),
            "random_seed": int(rng.integers(0, 1_000_000)),
        }


@dataclass(slots=True)
class SyntheticDatasetConfig:
    """Configuration for dataset generation."""

    image_shape: Tuple[int, int] = (96, 96)
    hyperspectral_bands: int = 32
    nir_scale: float = 1.0
    random_seed: int = 2025
    contamination_scale: float = 0.4
    background_level: float = 12.0


class SyntheticDatasetGenerator:
    """Produce ``MaterialSample`` instances from catalog-driven synthetic profiles."""

    def __init__(
        self,
        profiles: Sequence[SyntheticMaterialProfile],
        *,
        config: SyntheticDatasetConfig | None = None,
    ) -> None:
        if not profiles:
            raise ValueError("At least one profile is required")
        self.profiles = list(profiles)
        self.config = config or SyntheticDatasetConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def _profile_choice(self) -> SyntheticMaterialProfile:
        index = self.rng.integers(0, len(self.profiles))
        return self.profiles[int(index)]

    def _spectral_response(self, profile: SyntheticMaterialProfile) -> np.ndarray:
        record = profile.base_record()
        base = np.asarray(record["spectral_signature"], dtype=np.float32)
        jitter = self.rng.normal(scale=profile.spectral_jitter, size=base.shape)
        response = profile.spectral_gain * (base + jitter)
        response = np.clip(response, 0.0, None)
        return response

    def _simulate_rgb(self, profile: SyntheticMaterialProfile, spectral_response: np.ndarray) -> np.ndarray:
        height, width = self.config.image_shape
        grid_y, grid_x = np.meshgrid(
            np.linspace(0, 1, height, dtype=np.float32),
            np.linspace(0, 1, width, dtype=np.float32),
            indexing="ij",
        )
        base_pattern = spectral_response[:3].mean() * (0.6 + 0.4 * grid_x)
        modulation = spectral_response[3:6].mean() * (0.4 + 0.6 * grid_y)
        rgb = np.stack([
            base_pattern + modulation,
            base_pattern * 0.8 + modulation * 1.1,
            base_pattern * 1.2 + modulation * 0.7,
        ], axis=-1)
        rgb += self.config.background_level
        noise_scale = profile_noise_scale(spectral_response) * profile.noise_scale
        noise = self.rng.normal(scale=noise_scale, size=rgb.shape)
        rgb += noise
        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    def _simulate_nir(self, profile: SyntheticMaterialProfile, spectral_response: np.ndarray) -> np.ndarray:
        height, width = self.config.image_shape
        nir = spectral_response.mean() * np.ones((height, width), dtype=np.float32)
        gradient = np.linspace(0, 1, height, dtype=np.float32).reshape(-1, 1)
        nir *= (0.7 + 0.3 * gradient)
        nir += self.rng.normal(scale=3.0 * self.config.nir_scale * profile.noise_scale, size=nir.shape)
        return np.clip(nir, 0.0, 255.0).astype(np.uint8)

    def _simulate_hyperspectral(
        self, profile: SyntheticMaterialProfile, spectral_response: np.ndarray
    ) -> np.ndarray:
        bands = min(self.config.hyperspectral_bands, spectral_response.size)
        response = spectral_response[:bands]
        height, width = self.config.image_shape
        base_cube = response[:, None, None] * np.ones((bands, height, width), dtype=np.float32)
        spatial_texture = self.rng.normal(
            size=(bands, height, width), scale=0.05 * profile.noise_scale
        )
        hyperspectral = base_cube * (1.0 + spatial_texture)
        return hyperspectral

    def generate_sample(self) -> MaterialSample:
        profile = self._profile_choice()
        record = profile.base_record()
        spectral_response = self._spectral_response(profile)
        rgb = self._simulate_rgb(profile, spectral_response)
        nir = self._simulate_nir(profile, spectral_response)
        hyperspectral = self._simulate_hyperspectral(profile, spectral_response)
        contamination_probability = profile.contamination_chance * self.config.contamination_scale
        contamination_score = float(self.rng.beta(2.0, 5.0) * contamination_probability)
        recyclable_probability = float(record["recyclable"]) * profile.recyclability_bias
        recyclable = bool(self.rng.random() < recyclable_probability)
        quality = float(
            np.clip(
                profile.quality_baseline + self.rng.normal(scale=0.05) - contamination_score * 0.3,
                0.0,
                1.0,
            )
        )
        tensor = SpectralTensor(
            rgb=rgb,
            nir=nir,
            hyperspectral=hyperspectral.astype(np.float32),
            metadata=profile.sample_metadata(self.rng),
        )
        sample = MaterialSample(
            tensor=tensor,
            label=profile.label,
            recyclable=recyclable,
            quality_score=quality,
            contamination_score=contamination_score,
        )
        return sample

    def stream(self, count: int) -> Iterator[MaterialSample]:
        for _ in range(count):
            yield self.generate_sample()

    def build_dataset(self, count: int) -> MaterialDataset:
        return MaterialDataset(list(self.stream(count)))

    def export_npz(self, directory: Path, count: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(self.stream(count)):
            tensor = sample.tensor
            path = directory / f"sample_{index:03d}.npz"
            metadata = tensor.metadata
            metadata_json = json.dumps(metadata)
            np.savez_compressed(
                path,
                rgb=tensor.rgb,
                nir=tensor.nir,
                hyperspectral=tensor.hyperspectral,
                label=sample.label,
                recyclable=sample.recyclable,
                quality_score=sample.quality_score,
                contamination_score=sample.contamination_score,
                metadata=np.array(metadata_json, dtype=np.str_),
            )


def profile_noise_scale(spectral_response: np.ndarray) -> float:
    """Compute noise scale based on spectral dynamic range."""

    spread = spectral_response.max() - spectral_response.min()
    return float(np.clip(spread * 4.0, 2.0, 30.0))


def default_profiles() -> List[SyntheticMaterialProfile]:
    """Return curated synthetic profiles derived from the catalog."""

    curated_ids = [
        "AdvancedMaterial001",
        "AdvancedMaterial010",
        "AdvancedMaterial020",
        "AdvancedMaterial035",
        "AdvancedMaterial040",
        "AdvancedMaterial055",
        "AdvancedMaterial070",
        "AdvancedMaterial088",
        "AdvancedMaterial101",
        "AdvancedMaterial120",
    ]
    profiles: List[SyntheticMaterialProfile] = []
    for identifier in curated_ids:
        record = MATERIAL_DATABASE[identifier]
        profiles.append(
            SyntheticMaterialProfile(
                identifier=identifier,
                label=record["category"],
                noise_scale=0.8,
                contamination_chance=0.15,
                recyclability_bias=0.9 if record["recyclable"] else 0.4,
                quality_baseline=0.85 if record["recyclable"] else 0.6,
                spectral_gain=1.2,
                spectral_jitter=0.03,
            )
        )
    return profiles


def generator_from_catalog(
    labels: Sequence[str] | None = None,
    *,
    config: SyntheticDatasetConfig | None = None,
) -> SyntheticDatasetGenerator:
    """Create a generator using catalog entries filtered by labels."""

    profiles = []
    records = MATERIAL_DATABASE.items()
    if labels is not None:
        allowed = set(labels)
        records = [(name, record) for name, record in records if record["category"] in allowed]
    for name, record in records:
        profiles.append(
            SyntheticMaterialProfile(
                identifier=name,
                label=record["category"],
                noise_scale=0.6,
                contamination_chance=0.1,
                recyclability_bias=0.95 if record["recyclable"] else 0.5,
                quality_baseline=0.9 if record["recyclable"] else 0.55,
                spectral_gain=1.0,
                spectral_jitter=0.02,
            )
        )
    config = config or SyntheticDatasetConfig()
    return SyntheticDatasetGenerator(profiles, config=config)


def export_reference_dataset(root: Path, *, train_count: int = 120, val_count: int = 40) -> None:
    """Export a reference dataset to ``root`` using default profiles."""

    generator = SyntheticDatasetGenerator(default_profiles())
    generator.export_npz(root / "train", train_count)
    generator.export_npz(root / "val", val_count)


__all__ = [
    "SyntheticMaterialProfile",
    "SyntheticDatasetConfig",
    "SyntheticDatasetGenerator",
    "profile_noise_scale",
    "default_profiles",
    "generator_from_catalog",
    "export_reference_dataset",
]
