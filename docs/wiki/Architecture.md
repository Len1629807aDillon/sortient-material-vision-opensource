# Architecture

## Data Layer

- **SpectralTensor** encapsulates RGB, NIR, and hyperspectral streams.
- **MaterialDataset** and **DataLoader** orchestrate batched data access.
- **AugmentationPipeline** extends data variability with spatial and
  spectral perturbations.

## Model Layer

- **BACKBONES** registry centralises spectral feature extractors.
- **MultiSpectralClassifier** fuses backbone features into multi-task
  outputs: material logits, recyclability, contamination, and quality
  scores.

## Training Layer

- **Trainer** provides NumPy-backed optimisation, callback handling, and
  learning-rate scheduling (warmup + cosine decay).
- **MetricLogger** and **EarlyStopping** make experimentation traceable
  and efficient.

## Inference Layer

- **InferencePipeline** standardises sample normalisation, class
  post-processing, and JSON-compatible output generation.
- **FastAPI Service** (optional) exposes `/infer` for batch predictions.

## Sensor + Analytics

- **SensorFactory** & **calibration** support sensor simulation.
- **PerformanceMonitor** and console dashboard create operational
  telemetry views.
