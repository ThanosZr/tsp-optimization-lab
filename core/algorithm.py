"""The plugin contract every algorithm implements.

Design goals
------------
* The GUI never hard-codes algorithm names. It iterates over registered
  algorithms and talks to them only through the `Algorithm` interface below.
* Each algorithm declares its work as a **step graph** (`Step` with `requires`)
  so the UI can show dependencies and colour-code readiness.
* Cost is measured honestly through `CostMeter` (logical steps are NOT the same
  as computational work) so "5000 steps" stops being misleading.

Adding a new algorithm = subclass `Algorithm`, declare `STEPS`, implement the
step bodies, and register it. Nothing in the GUI needs editing. See
ARCHITECTURE.md.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from core.models import Node


# --------------------------------------------------------------------------- #
# Step graph
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Step:
    id: str
    label: str
    requires: List[str] = field(default_factory=list)   # step ids that must be DONE first
    produces: List[str] = field(default_factory=list)   # logical artefacts (docs/UI only)
    description: str = ""
    # If True the step can be re-run after completion (e.g. recompute exact).
    rerunnable: bool = False


# Status strings (kept as plain strings for trivial serialisation)
LOCKED = "locked"     # a prerequisite is not DONE yet
READY = "ready"       # prerequisites satisfied, not run yet
RUNNING = "running"   # currently animating
DONE = "done"         # completed


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #
@dataclass
class CostMeter:
    """Tracks *real* work, not just logical step counts."""
    logical_steps: int = 0      # algorithm-defined "ticks" (e.g. SOM city presentations)
    distance_evals: int = 0     # pairwise distance computations
    unit_updates: int = 0       # element updates (e.g. neuron moves)
    elapsed_ms: float = 0.0     # cumulative wall-clock time spent computing

    _t0: Optional[float] = field(default=None, repr=False, compare=False)

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._t0 is not None:
            self.elapsed_ms += (time.perf_counter() - self._t0) * 1000.0
            self._t0 = None

    def reset(self) -> None:
        self.logical_steps = 0
        self.distance_evals = 0
        self.unit_updates = 0
        self.elapsed_ms = 0.0
        self._t0 = None

    def to_dict(self) -> dict:
        return {
            "logical_steps": self.logical_steps,
            "distance_evals": self.distance_evals,
            "unit_updates": self.unit_updates,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CostMeter":
        m = cls()
        m.logical_steps = int(d.get("logical_steps", 0))
        m.distance_evals = int(d.get("distance_evals", 0))
        m.unit_updates = int(d.get("unit_updates", 0))
        m.elapsed_ms = float(d.get("elapsed_ms", 0.0))
        return m


# --------------------------------------------------------------------------- #
# Per-frame visual snapshot returned by micro_step / run_*
# --------------------------------------------------------------------------- #
@dataclass
class VisualFrame:
    algo_id: str
    step_id: str
    phase: str
    message: str
    highlight_nodes: List[int] = field(default_factory=list)
    highlight_edges: List = field(default_factory=list)   # list[(a, b)]
    notes: Dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Visual layer descriptor (drives the GUI's per-algorithm layer toggles)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LayerSpec:
    id: str
    label: str
    default_on: bool = True


# --------------------------------------------------------------------------- #
# Algorithm base class
# --------------------------------------------------------------------------- #
class Algorithm:
    """Base class. Subclasses MUST set `id`, `name`, `color`, `STEPS`."""

    id: str = "base"
    name: str = "Base"
    color: str = "#1f77b4"          # base colour family for this algorithm's overlays
    STEPS: List[Step] = []

    def __init__(self, nodes: List[Node], params: Dict[str, object]) -> None:
        self.nodes = nodes
        self.params = dict(params)
        self.cost = CostMeter()
        self._done_steps: set[str] = set()
        self.active_step_id: Optional[str] = self.STEPS[0].id if self.STEPS else None

    # ---- step graph introspection ---------------------------------------- #
    def steps(self) -> List[Step]:
        return list(self.STEPS)

    def _step(self, step_id: str) -> Step:
        for s in self.STEPS:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def status(self, step_id: str) -> str:
        step = self._step(step_id)
        if step_id in self._done_steps and not step.rerunnable:
            return DONE
        if any(r not in self._done_steps for r in step.requires):
            return LOCKED
        if step_id in self._done_steps:  # rerunnable + done -> still READY again
            return READY
        return READY

    def is_step_done(self, step_id: str) -> bool:
        return step_id in self._done_steps

    def is_done(self) -> bool:
        # "Done" means every step has completed at least once (rerunnable
        # included), so run_all reaches independent steps like exact.
        return len(self.STEPS) > 0 and all(s.id in self._done_steps for s in self.STEPS)

    def next_ready_step(self) -> Optional[str]:
        """First step not yet completed whose prerequisites are all done."""
        for s in self.STEPS:
            if s.id in self._done_steps:
                continue
            if all(r in self._done_steps for r in s.requires):
                return s.id
        return None

    # ---- lifecycle (override in subclasses) ------------------------------ #
    def reset(self) -> VisualFrame:
        self.cost.reset()
        self._done_steps.clear()
        self.active_step_id = self.STEPS[0].id if self.STEPS else None
        return self._frame("reset", "reset", "Algorithm reset.")

    def micro_step(self) -> Optional[VisualFrame]:
        """Advance the finest animatable unit. Return None when nothing to do."""
        raise NotImplementedError

    def run_step(self, step_id: str) -> List[VisualFrame]:
        """Run the given step to completion (must be READY). Default loops micro_step."""
        if self.status(step_id) == LOCKED:
            return [self._frame(step_id, "blocked",
                                f"'{self._step(step_id).label}' is locked; "
                                f"run prerequisites first.")]
        # re-running a rerunnable step: clear its completion so it runs again
        if self._step(step_id).rerunnable and step_id in self._done_steps:
            self._done_steps.discard(step_id)
        self.active_step_id = step_id
        frames: List[VisualFrame] = []
        guard = 0
        while not self.is_step_done(step_id) and guard < 5_000_000:
            f = self.micro_step()
            guard += 1
            if f is None:
                break
            frames.append(f)
            if f.phase == "blocked":
                break
            if f.step_id != step_id and self.is_step_done(step_id):
                break
        return frames

    def run_all(self) -> List[VisualFrame]:
        frames: List[VisualFrame] = []
        guard = 0
        while not self.is_done() and guard < 50:
            nxt = self.next_ready_step()
            if nxt is None:
                break
            frames.extend(self.run_step(nxt))
            if not self.is_step_done(nxt):
                break
            guard += 1
        return frames

    # ---- results / serialisation (override in subclasses) ---------------- #
    def result(self) -> dict:
        """JSON-serialisable result (tours, regions, scalar metrics ...)."""
        return {"done_steps": sorted(self._done_steps), "cost": self.cost.to_dict()}

    def load_result(self, data: dict) -> None:
        self._done_steps = set(data.get("done_steps", []))
        self.cost = CostMeter.from_dict(data.get("cost", {}))

    def summary_metrics(self) -> Dict[str, object]:
        """Row for the comparison table: {'length':..., 'tour':..., ...}."""
        return {}

    # ---- visual layers (override in subclasses) -------------------------- #
    def layers(self) -> List[LayerSpec]:
        return []

    def draw(self, ax, ctx: "DrawContext") -> None:  # noqa: F821
        """Render this algorithm's overlays onto a matplotlib Axes."""
        return None

    # ---- helpers --------------------------------------------------------- #
    def _frame(self, step_id, phase, message, **kw) -> VisualFrame:
        self.cost.logical_steps += 1
        return VisualFrame(algo_id=self.id, step_id=step_id, phase=phase,
                           message=message, **kw)

    def node_by_id(self) -> Dict[int, Node]:
        return {n.idx: n for n in self.nodes}


@dataclass
class DrawContext:
    """Everything an Algorithm needs to render itself, passed by the GUI."""
    by_id: Dict[int, Node]
    layer_on: Dict[str, bool]          # layer_id -> visible?
    emphasis: bool = False             # caller wants this algorithm emphasised
    z_base: int = 1
