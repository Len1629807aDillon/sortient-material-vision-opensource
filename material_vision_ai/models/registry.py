"""Generic registry utilities for model components."""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple registry mapping names to callables."""

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, builder: Callable[..., T]) -> None:
        if name in self._builders:
            raise ValueError(f"Builder {name} already registered")
        self._builders[name] = builder

    def create(self, name: str, **kwargs) -> T:
        if name not in self._builders:
            raise KeyError(f"No builder registered under {name}")
        return self._builders[name](**kwargs)

    def names(self) -> Iterator[str]:
        return iter(self._builders.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._builders
