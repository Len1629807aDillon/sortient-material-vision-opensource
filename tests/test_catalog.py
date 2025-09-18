from __future__ import annotations

import numpy as np

from material_vision_ai.catalog import MATERIAL_DATABASE
from material_vision_ai.catalog.registry import (
    build_feature_matrix,
    compute_category_summaries,
    compute_similarity_matrix,
    spectral_embedding,
    top_similar_materials,
)


def test_catalog_summary_statistics() -> None:
    summaries = compute_category_summaries()
    assert summaries
    for summary in summaries[:5]:
        assert summary.average_density > 0
        assert summary.average_melting_point > 0


def test_similarity_matrix_shape() -> None:
    similarity = compute_similarity_matrix()
    assert similarity.shape[0] == similarity.shape[1]
    assert np.allclose(np.diag(similarity), 1.0)


def test_spectral_embedding_dimensions() -> None:
    names = list(MATERIAL_DATABASE.keys())[:20]
    embedding = spectral_embedding(names, components=2)
    assert set(embedding.keys()) == set(names)
    for vector in embedding.values():
        assert vector.shape == (2,)


def test_feature_matrix_construction() -> None:
    matrix = build_feature_matrix(["density", "melting_point_celsius"])
    assert matrix.ndim == 2
    assert matrix.shape[1] == 2


def test_similarity_results_ordering() -> None:
    reference = next(iter(MATERIAL_DATABASE.keys()))
    results = top_similar_materials(reference, k=3)
    assert len(results) == 3
    assert all(results[i].similarity >= results[i + 1].similarity for i in range(2))
