"""Point-cloud generators. Each is registered so the GUI/CLI pick them up
automatically. Add a new distribution = one decorated function."""
from __future__ import annotations

from typing import List

import numpy as np

from core.models import Node
from core.registry import register_generator


@register_generator("uniform")
def _uniform(n: int, seed: int) -> List[Node]:
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2))
    return [Node(i, float(x), float(y)) for i, (x, y) in enumerate(xy)]


@register_generator("clustered")
def _clustered(n: int, seed: int) -> List[Node]:
    rng = np.random.default_rng(seed)
    centers = rng.random((max(2, n // 15), 2))
    pts = []
    for _ in range(n):
        c = centers[rng.integers(0, len(centers))]
        pts.append(np.clip(c + rng.normal(0, 0.055, size=2), 0.0, 1.0))
    xy = np.array(pts) if pts else np.zeros((0, 2))
    return [Node(i, float(x), float(y)) for i, (x, y) in enumerate(xy)]


@register_generator("circle")
def _circle(n: int, seed: int) -> List[Node]:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, max(n, 1), endpoint=False) + rng.normal(0, 0.035, size=max(n, 1))
    radii = 0.38 + rng.normal(0, 0.025, size=max(n, 1))
    x = np.clip(0.5 + radii * np.cos(angles), 0.0, 1.0)
    y = np.clip(0.5 + radii * np.sin(angles), 0.0, 1.0)
    xy = np.stack([x, y], axis=1)[:n]
    return [Node(i, float(px), float(py)) for i, (px, py) in enumerate(xy)]


def make_nodes(n: int, seed: int = 7, mode: str = "uniform") -> List[Node]:
    from core.registry import get_generator
    if n <= 0:
        return []
    return get_generator(mode)(n, seed)
