"""Visualization utilities for spectral data."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - fallback when matplotlib is unavailable
    plt = None  # type: ignore

from ..data.dataset import MaterialSample


def plot_spectral_signature(sample: MaterialSample, bands: Iterable[int] | None = None) -> Tuple[plt.Figure, plt.Axes]:
    if plt is None:  # pragma: no cover - environment fallback
        raise RuntimeError("matplotlib is required for plotting")
    normalized = sample.tensor.normalized()
    spectrum = normalized.hyperspectral
    if bands is not None:
        spectrum = spectrum[np.asarray(list(bands))]
    mean_signature = spectrum.mean(axis=(1, 2))
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(mean_signature)
    axis.set_title(f"Spectral signature for {sample.label}")
    axis.set_xlabel("Band index")
    axis.set_ylabel("Normalized intensity")
    return figure, axis
