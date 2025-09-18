"""FastAPI service exposing the inference pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover - fallback when FastAPI is unavailable
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    BaseModel = object  # type: ignore

from ..data.dataset import MaterialSample, SpectralTensor
from ..utils.logging import get_logger
from .pipeline import InferencePipeline

logger = get_logger(__name__)


class SampleRequest(BaseModel):  # type: ignore[misc]
    rgb: List[List[List[float]]]
    nir: List[List[float]]
    hyperspectral: List[List[List[float]]]
    metadata: Dict[str, Any] | None = None
    label: str | None = None
    recyclable: bool | None = None
    quality_score: float | None = None
    contamination_score: float | None = None


class InferenceRequest(BaseModel):  # type: ignore[misc]
    samples: List[SampleRequest]


class InferenceResponse(BaseModel):  # type: ignore[misc]
    predictions: List[Dict[str, float | str]]


def build_app(config: Dict[str, Any]) -> FastAPI:
    if FastAPI is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("FastAPI is required to build the inference service")
    pipeline = InferencePipeline.from_config(config)
    app = FastAPI(title="Material Vision AI", version="0.1.0")

    @app.post("/infer", response_model=InferenceResponse)
    def infer(request: InferenceRequest) -> InferenceResponse:
        if not request.samples:
            raise HTTPException(status_code=400, detail="No samples provided")
        samples: List[MaterialSample] = []
        for sample in request.samples:
            tensor = SpectralTensor(
                rgb=np.asarray(sample.rgb, dtype=float),
                nir=np.asarray(sample.nir, dtype=float),
                hyperspectral=np.asarray(sample.hyperspectral, dtype=float),
                metadata=sample.metadata or {},
            )
            samples.append(
                MaterialSample(
                    tensor=tensor,
                    label=sample.label or "unknown",
                    recyclable=bool(sample.recyclable) if sample.recyclable is not None else True,
                    quality_score=float(sample.quality_score or 0.0),
                    contamination_score=float(sample.contamination_score or 0.0),
                )
            )
        predictions = pipeline.predict_samples(samples)
        return InferenceResponse(predictions=predictions)

    return app
