"""Command line interface for Material Vision AI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .inference.pipeline import InferencePipeline
from .training.trainer import Trainer
from .utils.config import load_config_file
from .utils.logging import configure_logging


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="material-vision-ai",
        description=(
            "Command line interface for Sortient's Material Vision AI "
            "platform. Provides quick entry points for training, "
            "evaluation, and inference workflows."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a YAML or JSON configuration file describing the pipeline.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        required=True,
        help="Specifies whether to run training or inference workflows.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional path to an input sample or dataset manifest for inference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for storing output artifacts such as checkpoints or predictions.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, steps are logged without executing heavy computation. Useful for debugging configs.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    configure_logging(level=args.log_level)
    config = load_config_file(args.config) if args.config else {}

    if args.mode == "train":
        trainer = Trainer.from_config(config)
        trainer.fit(dry_run=args.dry_run)
    elif args.mode == "infer":
        pipeline = InferencePipeline.from_config(config)
        results = pipeline.run(args.input, dry_run=args.dry_run)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2))
        else:
            print(json.dumps(results, indent=2))
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":  # pragma: no cover
    main()
