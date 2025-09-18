"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

import numpy as np

from .augmentations import AugmentationPipeline
from .dataset import MaterialDataset, MaterialSample
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Simple iterable over batches."""

    def __init__(
        self,
        dataset: MaterialDataset,
        batch_size: int,
        shuffle: bool = True,
        augmentation: Optional[AugmentationPipeline] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation

    def __iter__(self) -> Iterator[List[MaterialSample]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        batch: List[MaterialSample] = []
        for idx in indices:
            sample = self.dataset.at(idx)
            if self.augmentation:
                sample = self.augmentation.transform(sample)
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def build_dataloaders(
    data_config: Dict[str, object],
    augmentation_factory: Callable[[], AugmentationPipeline] | None = None,
) -> Dict[str, DataLoader]:
    """Construct dataloaders from configuration."""

    dataset_root = Path(data_config.get("root", "examples/data"))
    batch_size = int(data_config.get("batch_size", 8))
    train_dataset = MaterialDataset.from_directory(dataset_root / "train")
    val_dataset = MaterialDataset.from_directory(dataset_root / "val")
    augmentation = augmentation_factory() if augmentation_factory else None
    logger.info(
        "Instantiated dataloaders root=%s batch_size=%d augmentation=%s",
        dataset_root,
        batch_size,
        augmentation_factory.__name__ if augmentation_factory else None,
    )
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, augmentation=augmentation),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, augmentation=None),
    }
