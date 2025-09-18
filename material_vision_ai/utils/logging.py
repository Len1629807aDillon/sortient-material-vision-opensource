"""Logging helpers for Material Vision AI."""

from __future__ import annotations

import logging
import os
from typing import Optional

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def configure_logging(level: str = "INFO", name: Optional[str] = None) -> None:
    """Configure logging for the package."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_DEFAULT_FORMAT,
    )
    if name:
        logger = logging.getLogger(name)
        logger.setLevel(level.upper())
    os.environ.setdefault("MATERIAL_VISION_LOG_LEVEL", level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""

    level = os.environ.get("MATERIAL_VISION_LOG_LEVEL", "INFO")
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
