"""Tiny decorator-based registries.

Register an algorithm / data generator / exporter once and it shows up
everywhere (GUI dropdowns, CLI, export menu) with zero further wiring.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Type

# ------------------------------- algorithms -------------------------------- #
_ALGORITHMS: Dict[str, Type] = {}


def register_algorithm(cls: Type) -> Type:
    if not getattr(cls, "id", None):
        raise ValueError("Algorithm subclass must define a unique `id`.")
    _ALGORITHMS[cls.id] = cls
    return cls


def algorithm_classes() -> List[Type]:
    return list(_ALGORITHMS.values())


def get_algorithm(algo_id: str) -> Type:
    return _ALGORITHMS[algo_id]


# ---------------------------- data generators ------------------------------ #
# name -> callable(n, seed) -> list[Node]
_GENERATORS: Dict[str, Callable] = {}


def register_generator(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        _GENERATORS[name] = fn
        return fn
    return deco


def generator_names() -> List[str]:
    return list(_GENERATORS.keys())


def get_generator(name: str) -> Callable:
    return _GENERATORS[name]


# -------------------------------- exporters -------------------------------- #
# label -> callable(session, path) -> None ; `ext` is the default file extension
_EXPORTERS: Dict[str, dict] = {}


def register_exporter(label: str, ext: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        _EXPORTERS[label] = {"fn": fn, "ext": ext}
        return fn
    return deco


def exporters() -> Dict[str, dict]:
    return dict(_EXPORTERS)
