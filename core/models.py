"""Core data structures shared across the workbench.

This module is intentionally framework-free (no tkinter / matplotlib import) so
it can be reused by the GUI, the CLI runner and tests alike.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Node:
    """An immutable 2D point in the unit square (idx is its stable id)."""
    idx: int
    x: float
    y: float


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def distance(a: Node, b: Node) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def tour_edges(order: List[int]) -> List[Tuple[int, int]]:
    """Closed-cycle edge list for an ordered list of node ids."""
    if len(order) < 2:
        return []
    edges = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    edges.append((order[-1], order[0]))
    return edges


def tour_length(nodes: List[Node], order: List[int]) -> float:
    by_id = {n.idx: n for n in nodes}
    total = 0.0
    for a, b in tour_edges(order):
        total += math.hypot(by_id[a].x - by_id[b].x, by_id[a].y - by_id[b].y)
    return total


# --------------------------------------------------------------------------- #
# Generic, algorithm-agnostic result container
# --------------------------------------------------------------------------- #
@dataclass
class TourResult:
    """A named TSP tour produced by some algorithm step.

    `kind` distinguishes overlay styling in the GUI (e.g. 'heuristic',
    'refined', 'optimal'). `meta` is free-form JSON-serialisable extra data.
    """
    label: str
    order: List[int] = field(default_factory=list)
    length: float = 0.0
    kind: str = "heuristic"
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return tour_edges(self.order)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "order": list(self.order),
            "length": self.length,
            "kind": self.kind,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TourResult":
        return cls(
            label=d["label"],
            order=list(d.get("order", [])),
            length=float(d.get("length", 0.0)),
            kind=d.get("kind", "heuristic"),
            meta=dict(d.get("meta", {})),
        )
