"""Generate a catalog-aligned synthetic multi-spectral dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from material_vision_ai.data.synthetic import (
    SyntheticDatasetConfig,
    SyntheticDatasetGenerator,
    default_profiles,
    export_reference_dataset,
    generator_from_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory where the dataset will be written")
    parser.add_argument("--train-count", type=int, default=240, help="Number of training samples")
    parser.add_argument("--val-count", type=int, default=80, help="Number of validation samples")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of catalog categories to include",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=(96, 96), help="Image height and width")
    parser.add_argument("--bands", type=int, default=32, help="Number of hyperspectral bands")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed controlling reproducibility")
    return parser.parse_args()


def build_generator(categories: Sequence[str] | None, args: argparse.Namespace) -> SyntheticDatasetGenerator:
    config = SyntheticDatasetConfig(
        image_shape=tuple(args.image_size),
        hyperspectral_bands=args.bands,
        random_seed=args.seed,
    )
    if categories:
        return generator_from_catalog(categories, config=config)
    return SyntheticDatasetGenerator(default_profiles(), config=config)


def main() -> None:
    args = parse_args()
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.categories is None:
        export_reference_dataset(output, train_count=args.train_count, val_count=args.val_count)
        print(f"Reference dataset exported to {output}")
    else:
        generator = build_generator(args.categories, args)
        generator.export_npz(output / "train", args.train_count)
        generator.export_npz(output / "val", args.val_count)
        print(
            "Custom dataset exported to",
            output,
            "categories=",
            ",".join(args.categories),
        )


if __name__ == "__main__":
    main()
