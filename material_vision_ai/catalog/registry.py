"""Analytical accessors for the material catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .database import MATERIAL_DATABASE, MaterialRecord


@dataclass(slots=True)
class CategorySummary:
    """Statistical summary for a material category."""

    category: str
    members: List[str]
    average_density: float
    average_melting_point: float
    average_co2: float
    mean_spectral_signature: np.ndarray
    sustainability_index: float

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "category": self.category,
            "count": len(self.members),
            "average_density": float(self.average_density),
            "average_melting_point": float(self.average_melting_point),
            "average_co2": float(self.average_co2),
            "sustainability_index": float(self.sustainability_index),
        }


@dataclass(slots=True)
class SimilarityResult:
    """Represents pairwise similarity between two materials."""

    source: str
    target: str
    similarity: float
    distance: float


def _spectral_vector(record: MaterialRecord) -> np.ndarray:
    return np.asarray(record["spectral_signature"], dtype=np.float32)


def compute_category_summaries() -> List[CategorySummary]:
    """Aggregate catalog records into category-level summaries."""

    by_category: Dict[str, List[str]] = {}
    for name, record in MATERIAL_DATABASE.items():
        by_category.setdefault(record["category"], []).append(name)
    summaries: List[CategorySummary] = []
    for category, members in by_category.items():
        records = [MATERIAL_DATABASE[name] for name in members]
        densities = np.array([record["density"] for record in records], dtype=np.float64)
        melting_points = np.array([record["melting_point_celsius"] for record in records], dtype=np.float64)
        co2 = np.array([record["co2_per_kg"] for record in records], dtype=np.float64)
        sustainability = np.array([record["sustainability_index"] for record in records], dtype=np.float64)
        signature_matrix = np.vstack([_spectral_vector(record) for record in records])
        summary = CategorySummary(
            category=category,
            members=sorted(members),
            average_density=float(densities.mean()),
            average_melting_point=float(melting_points.mean()),
            average_co2=float(co2.mean()),
            mean_spectral_signature=signature_matrix.mean(axis=0),
            sustainability_index=float(sustainability.mean()),
        )
        summaries.append(summary)
    summaries.sort(key=lambda summary: summary.category)
    return summaries


def compute_similarity_matrix(materials: Sequence[str] | None = None) -> np.ndarray:
    """Compute cosine similarity matrix for selected materials."""

    if materials is None:
        materials = list(MATERIAL_DATABASE.keys())[:50]
    matrix = np.vstack([_spectral_vector(MATERIAL_DATABASE[name]) for name in materials])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, a_min=1e-9, a_max=None)
    return normalized @ normalized.T


def top_similar_materials(reference: str, *, k: int = 5) -> List[SimilarityResult]:
    """Return the top ``k`` most similar materials to ``reference``."""

    if reference not in MATERIAL_DATABASE:
        raise KeyError(f"Unknown material '{reference}'")
    materials = [name for name in MATERIAL_DATABASE.keys() if name != reference]
    ref_vector = _spectral_vector(MATERIAL_DATABASE[reference])
    ref_norm = np.linalg.norm(ref_vector)
    results: List[SimilarityResult] = []
    for name in materials:
        vector = _spectral_vector(MATERIAL_DATABASE[name])
        norm = np.linalg.norm(vector)
        similarity = float(np.dot(ref_vector, vector) / (ref_norm * norm + 1e-9))
        distance = float(np.linalg.norm(ref_vector - vector))
        results.append(SimilarityResult(source=reference, target=name, similarity=similarity, distance=distance))
    results.sort(key=lambda result: result.similarity, reverse=True)
    return results[:k]


def spectral_embedding(materials: Sequence[str] | None = None, *, components: int = 3) -> Dict[str, np.ndarray]:
    """Embed material spectra using principal component analysis."""

    if materials is None:
        materials = list(MATERIAL_DATABASE.keys())
    matrix = np.vstack([_spectral_vector(MATERIAL_DATABASE[name]) for name in materials])
    matrix -= matrix.mean(axis=0, keepdims=True)
    covariance = matrix.T @ matrix / (matrix.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:components]
    projection = matrix @ eigenvectors[:, order]
    return {name: projection[idx] for idx, name in enumerate(materials)}


def metric_rankings(metric: str, *, top_k: int = 10, reverse: bool = False) -> List[Tuple[str, float]]:
    """Rank materials by a numeric metric present in the catalog."""

    if not MATERIAL_DATABASE:
        return []
    if metric not in next(iter(MATERIAL_DATABASE.values())):
        raise KeyError(f"Metric '{metric}' not present in catalog")
    values = [
        (name, float(record[metric]))
        for name, record in MATERIAL_DATABASE.items()
        if isinstance(record.get(metric), (int, float))
    ]
    values.sort(key=lambda pair: pair[1], reverse=reverse)
    return values[:top_k]


def build_feature_matrix(metrics: Sequence[str]) -> np.ndarray:
    """Construct a feature matrix for downstream machine learning tasks."""

    matrix: List[List[float]] = []
    for record in MATERIAL_DATABASE.values():
        row = []
        for metric in metrics:
            value = record.get(metric, 0.0)
            if isinstance(value, (int, float)):
                row.append(float(value))
            else:
                raise TypeError(f"Metric '{metric}' is not numeric for record")
        matrix.append(row)
    return np.asarray(matrix, dtype=np.float32)


def select_recyclable_by_category(category: str) -> Dict[str, MaterialRecord]:
    """Return recyclable materials for a given category."""

    output: Dict[str, MaterialRecord] = {}
    for name, record in MATERIAL_DATABASE.items():
        if record["category"] == category and record["recyclable"]:
            output[name] = record
    return output
