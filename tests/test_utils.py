from pathlib import Path

from material_vision_ai.utils.config import dump_config, load_config_file


def test_config_roundtrip(tmp_path: Path) -> None:
    config = {"name": "demo", "model": {"num_classes": 4}}
    path = tmp_path / "config.yaml"
    dump_config(config, path)
    loaded = load_config_file(path)
    assert loaded["name"] == "demo"
    assert loaded["model"]["num_classes"] == 4
