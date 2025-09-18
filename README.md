# Material Vision AI

Material Vision AI is an open-source initiative by Sortient to share the
core technology that powers our industrial recycling detection systems.
The project focuses on advanced multi-spectrum imaging, AI-driven
material classification, contamination analysis, and recyclability
assessment – bringing research-grade experimentation and
production-oriented tooling together in one cohesive platform.

## Key Highlights

- **Multi-spectrum fusion**: RGB, NIR, and hyperspectral streams are
  fused into robust feature representations designed for demanding
  recycling environments.
- **Hybrid training pipeline**: Lightweight NumPy-based optimisation for
  rapid experimentation with optional PyTorch backbones for GPU-accelerated
  research.
- **Recyclability intelligence**: Built-in targets for recyclability
  scores, contamination prediction, and quality grading alongside the
  primary material classification head.
- **Sensor-aware simulation**: Utilities for modelling RGB, NIR, and
  hyperspectral sensors including noise simulation and calibration
  routines.
- **Modern inference tooling**: Turn-key inference pipeline, FastAPI
  service generator, configurable CLI, and analytics dashboards that make
  it easy to embed Material Vision AI in production systems.
- **Synthetic reference dataset**: A reproducible generator for
  multi-spectral samples to facilitate experimentation without requiring
  proprietary data.

## Repository Layout

```
material_vision_ai/
  analytics/          # Monitoring utilities and console dashboard renderer
  catalog/            # 200+ material records with spectral properties
  data/               # Dataset abstractions, loaders, augmentations, synthesis
  evaluation/         # Metrics and reporting helpers
  inference/          # Inference pipeline and optional FastAPI service
  models/             # Backbone registry and multispectral classifier
  sensors/            # Sensor simulation and calibration helpers
  training/           # Configurable trainer with callbacks and schedulers
  utils/              # Logging and configuration utilities
  visualization/      # Spectral plotting with optional matplotlib backend
examples/
  configs/            # Ready-to-use experiment configurations
scripts/
  generate_synthetic_dataset.py
                       # Script to regenerate the synthetic dataset
tests/                 # Pytest suite covering key functionality
```

## Getting Started

### 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # alternatively install numpy/pyyaml manually
```

The package is compatible with Python 3.10+ and only requires NumPy and
PyYAML for the core experience. Optional extras include FastAPI (for the
web service) and matplotlib (for plotting utilities).

### 2. Generate the Synthetic Dataset

A compact synthetic dataset can be generated on demand. Choose an output
directory (for example `artifacts/dataset`) and run:

```bash
python scripts/generate_synthetic_dataset.py artifacts/dataset
```

The script leverages the catalog of 200+ materials to synthesise
multi-spectral samples aligned with Sortient's production systems. Use
`--categories` to restrict the output to specific material categories.

### 3. Run the Training Pipeline

```bash
python -m material_vision_ai.cli --mode train --config examples/configs/default.yaml
```

The trainer uses a NumPy-based optimisation routine by default, making it
suitable for CPU-only environments while still providing a path to
PyTorch backbones for accelerated workloads.

### 4. Perform Inference

```bash
python -m material_vision_ai.cli --mode infer --config examples/configs/default.yaml --input artifacts/dataset/val
```

The command outputs predictions including material labels, recyclability
estimates, contamination scores, and quality grades.

### 5. Launch the Optional FastAPI Service

```python
from material_vision_ai.inference import build_app
from material_vision_ai.utils import load_config_file

config = load_config_file(Path("examples/configs/default.yaml"))
app = build_app(config)
```

The resulting FastAPI app exposes an `/infer` endpoint accepting batches
of multi-spectral samples. FastAPI and Pydantic are optional
dependencies; install them via `pip install fastapi pydantic` if you plan
on using the service component.

## Advanced Features

- **Sensor calibration**: `material_vision_ai.sensors.calibration` offers
  bias/gain estimation routines to calibrate simulated or real sensor
  streams.
- **Analytics and monitoring**: `PerformanceMonitor` aggregates streaming
  metrics, and `render_console_dashboard` renders a ready-made console
  dashboard for live systems.
- **Visualization**: Optional integration with matplotlib enables rapid
  spectral signature plotting.
- **Comprehensive material catalog**: `material_vision_ai.catalog` bundles
  200+ reference materials with spectral signatures, thermal properties,
  and sustainability metrics to drive synthetic data generation and
  similarity analysis.
- **Extensible registries**: Model components are registered via the
  built-in `BACKBONES` registry, making it straightforward to add custom
  feature extractors.

## Testing

The repository ships with an automated test suite:

```bash
pytest
```

The tests validate dataset loading, training loops, inference pipelines,
analytics, and configuration utilities. They are lightweight enough to
run in continuous integration pipelines or local developer environments.

## Contributing

We welcome contributions from the research and industrial communities.
Please open issues for questions, bug reports, or feature requests, and
submit pull requests that align with the project's coding standards.

## License

Material Vision AI is released under the MIT License. See
[LICENSE](LICENSE) for details.
