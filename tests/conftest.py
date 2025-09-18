from __future__ import annotations

from pathlib import Path

import pytest

from material_vision_ai.data.synthetic import export_reference_dataset


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("material_dataset")
    export_reference_dataset(root, train_count=30, val_count=10)
    return root
