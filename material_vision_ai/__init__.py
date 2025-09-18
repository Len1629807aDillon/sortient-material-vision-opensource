"""Material Vision AI package.

This package provides tools for multi-spectrum material detection,
recyclability classification, contamination analysis, and quality
assessment. It is designed to offer both research-grade
experimentation utilities and production-ready inference pipelines.
"""

from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "__version__",
]

try:  # pragma: no cover - runtime environment may not have package metadata
    __version__ = version("material-vision-ai")
except PackageNotFoundError:  # pragma: no cover - when running from source
    __version__ = "0.1.0"
