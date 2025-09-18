from pathlib import Path

from material_vision_ai.training.trainer import Trainer
from material_vision_ai.inference.pipeline import InferencePipeline
from material_vision_ai.data.dataset import MaterialDataset


def build_config(dataset_root: Path) -> dict:
    dataset = MaterialDataset.from_directory(dataset_root / "train")
    class_names = sorted({sample.label for sample in dataset})
    return {
        "data": {"root": str(dataset_root), "batch_size": 4, "augment": False},
        "model": {
            "backbone": "numpy_spectral",
            "num_classes": len(class_names),
            "class_names": class_names,
        },
        "training": {"epochs": 2, "lr": 0.01, "scheduler": False, "early_stopping": False},
    }


def test_training_loop_runs(synthetic_dataset: Path) -> None:
    config = build_config(synthetic_dataset)
    trainer = Trainer.from_config(config)
    trainer.fit()
    assert "train_loss" in trainer.metrics
    assert trainer.metrics["train_loss"] >= 0.0


def test_inference_pipeline_predictions(synthetic_dataset: Path) -> None:
    dataset = MaterialDataset.from_directory(Path(synthetic_dataset) / "val")
    sample = next(iter(dataset))
    config = build_config(synthetic_dataset)
    pipeline = InferencePipeline.from_config(config)
    predictions = pipeline.predict_samples([sample])
    assert predictions
    assert "label" in predictions[0]
    assert "confidence" in predictions[0]
