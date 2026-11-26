"""A Session bundles one problem instance with every registered algorithm and
their results, and serialises the whole thing to JSON for save/load.

This is the single object the GUI and CLI both operate on.
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Dict, List, Optional

from core.algorithm import Algorithm
from core.models import Node
from core.registry import algorithm_classes

SESSION_VERSION = 4


class Session:
    def __init__(self, nodes: List[Node], seed: int = 7, data_mode: str = "uniform",
                 params: Optional[Dict[str, dict]] = None) -> None:
        self.nodes = nodes
        self.seed = seed
        self.data_mode = data_mode
        self.params: Dict[str, dict] = params or {}
        self.history: List[str] = []
        self.algorithms: Dict[str, Algorithm] = {}
        self._build_algorithms()

    # ------------------------------------------------------------------ #
    def _build_algorithms(self) -> None:
        self.algorithms = {}
        for cls in algorithm_classes():
            self.algorithms[cls.id] = cls(self.nodes, self.params.get(cls.id, {}))

    def rebuild_algorithms(self, params: Optional[Dict[str, dict]] = None) -> None:
        """Recreate fresh algorithm instances (e.g. after nodes/params change)."""
        if params is not None:
            self.params = params
        self._build_algorithms()

    def set_nodes(self, nodes: List[Node]) -> None:
        self.nodes = nodes
        self._build_algorithms()

    def log(self, msg: str) -> None:
        self.history.append(msg)
        if len(self.history) > 500:
            self.history = self.history[-500:]

    def algos(self) -> List[Algorithm]:
        return list(self.algorithms.values())

    # ------------------------------------------------------------------ #
    def comparison_rows(self) -> List[dict]:
        rows: List[dict] = []
        for algo in self.algos():
            rows.extend(algo.comparison_rows())
        # attach % gap vs the best 'optimal' length if one exists
        optimal = [r["length"] for r in rows if r.get("kind") == "optimal" and r.get("length")]
        ref = min(optimal) if optimal else None
        for r in rows:
            L = r.get("length")
            r["gap_pct"] = (100.0 * (L - ref) / ref) if (ref and L) else None
        return rows

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "version": SESSION_VERSION,
            "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "instance": {
                "n": len(self.nodes),
                "seed": self.seed,
                "data_mode": self.data_mode,
                "nodes": [[n.x, n.y] for n in self.nodes],
            },
            "params": self.params,
            "results": {aid: a.result() for aid, a in self.algorithms.items()},
            "history": self.history[-200:],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        inst = d.get("instance", {})
        nodes = [Node(i, float(x), float(y)) for i, (x, y) in enumerate(inst.get("nodes", []))]
        s = cls(nodes, seed=int(inst.get("seed", 7)),
                data_mode=inst.get("data_mode", "uniform"),
                params=d.get("params", {}))
        s.history = list(d.get("history", []))
        for aid, res in d.get("results", {}).items():
            if aid in s.algorithms:
                s.algorithms[aid].load_result(res)
        return s

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Session":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
