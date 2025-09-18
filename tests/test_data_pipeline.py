from pathlib import Path

from material_vision_ai.data.dataset import MaterialDataset
from material_vision_ai.data.loader import build_dataloaders
from material_vision_ai.data.augmentations import build_standard_augmentations


def test_dataset_loading(synthetic_dataset: Path) -> None:
    dataset = MaterialDataset.from_directory(synthetic_dataset / "train")
    assert len(dataset) > 0
    sample = next(iter(dataset))
    tensor = sample.tensor
    tensor.validate()
    normalized = tensor.normalized()
    assert normalized.rgb.max() <= 1.0


def test_dataloader_build(synthetic_dataset: Path) -> None:
    config = {"root": str(synthetic_dataset), "batch_size": 4, "augment": True}
    dataloaders = build_dataloaders(config, augmentation_factory=build_standard_augmentations)
    batch = next(iter(dataloaders["train"]))
    assert len(batch) == 4
