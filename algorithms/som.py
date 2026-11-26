"""SOM ring heuristic for Euclidean TSP — the visual *baseline / reference*.

Rebuilt on the `Algorithm` plugin contract:
  init -> train -> build_tour -> two_opt        (dependency chain)
  exact                                          (independent optimal reference)

The update maths are identical to the original v3 implementation, only
numpy-vectorised so the per-step cost (neuron updates + distance evals) is both
faster and *honestly counted* by the CostMeter.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import numpy as np

from algorithms.tsp_utils import held_karp_exact, two_opt
from core.algorithm import (Algorithm, DrawContext, LayerSpec, Step, VisualFrame)
from core.models import Node, TourResult, tour_length
from core.registry import register_algorithm


@register_algorithm
class SOMAlgorithm(Algorithm):
    id = "som"
    name = "SOM baseline"
    color = "#1f77b4"      # blue family

    STEPS = [
        Step("init", "Init ring", requires=[], produces=["neurons"],
             description="Place a circular ring of neurons around the cloud."),
        Step("train", "Train", requires=["init"], produces=["trained_neurons"],
             description="Present cities; pull the ring toward the data (SOM)."),
        Step("build_tour", "Build tour", requires=["train"], produces=["som_tour"],
             description="Order cities by their best-matching neuron on the ring."),
        Step("two_opt", "2-opt refine", requires=["build_tour"], produces=["opt_tour"],
             description="Local 2-opt improvement of the SOM tour."),
        Step("exact", "Exact (Held-Karp)", requires=[], produces=["exact_tour"],
             description="Optimal tour for small n. Independent reference.",
             rerunnable=True),
    ]

    def __init__(self, nodes: List[Node], params: Dict[str, object]) -> None:
        super().__init__(nodes, params)
        self.rng = random.Random(int(params.get("seed", 7)))
        self.neurons: Optional[np.ndarray] = None          # (m, 2)
        self.iteration = 0
        self.max_iterations = int(params.get("som_iterations", max(2000, len(nodes) * 150)))
        self.initial_lr = float(params.get("learning_rate", 0.8))
        self.initial_radius = 0.0
        self.lr = self.initial_lr
        self.radius = 0.0
        self.bmu_index: Optional[int] = None
        self.current_city: Optional[int] = None
        self.city_order: List[int] = []
        self.order_pos = 0
        self.tours: Dict[str, TourResult] = {}             # 'som' | 'opt' | 'exact'
        self.exact_message = ""

    # ------------------------------------------------------------------ #
    def reset(self) -> VisualFrame:
        super().reset()
        self.rng = random.Random(int(self.params.get("seed", 7)))
        self.neurons = None
        self.iteration = 0
        self.bmu_index = None
        self.current_city = None
        self.city_order = []
        self.order_pos = 0
        self.tours = {}
        self.exact_message = ""
        return self._frame("init", "reset", "SOM reset.")

    # ------------------------------------------------------------------ #
    def _do_init(self) -> VisualFrame:
        self.cost.start()
        pts = np.array([[p.x, p.y] for p in self.nodes], dtype=float)
        n = len(self.nodes)
        factor = float(self.params.get("neuron_factor", 8.0))
        m = int(self.params.get("neuron_count", max(n * 2, int(n * factor))))
        m = max(m, 4)
        self.max_iterations = int(self.params.get("som_iterations", max(2000, n * 150)))
        self.initial_lr = float(self.params.get("learning_rate", 0.8))
        self.initial_radius = float(self.params.get("radius", 0.0)) or max(2.0, m / 10)

        c = pts.mean(axis=0)
        spread = max(float(np.max(np.abs(pts - c))), 0.15)
        r = min(0.48, spread * 1.5)
        theta = 2 * np.pi * np.arange(m) / m
        self.neurons = np.stack([c[0] + r * np.cos(theta), c[1] + r * np.sin(theta)], axis=1)

        self.iteration = 0
        self.lr = self.initial_lr
        self.radius = self.initial_radius
        self.city_order = list(range(n))
        self.rng.shuffle(self.city_order)
        self.order_pos = 0
        self.cost.stop()
        self._done_steps.add("init")
        return self._frame("init", "init",
                           f"SOM initialised: {m} neurons, {self.max_iterations} iterations.",
                           notes={"neurons": str(m), "iterations": str(self.max_iterations)})

    def _train_one(self) -> VisualFrame:
        assert self.neurons is not None
        self.cost.start()
        if self.order_pos >= len(self.city_order):
            self.rng.shuffle(self.city_order)
            self.order_pos = 0
        city_idx = self.city_order[self.order_pos]
        self.order_pos += 1
        city = self.nodes[city_idx]
        cxy = np.array([city.x, city.y])

        t = self.iteration / max(1, self.max_iterations)
        self.lr = self.initial_lr * math.exp(-3.0 * t)
        self.radius = max(self.initial_radius * math.exp(-3.0 * t), 0.5)

        m = len(self.neurons)
        diff = self.neurons - cxy
        d2 = np.einsum("ij,ij->i", diff, diff)          # m distance evals (squared)
        bmu = int(np.argmin(d2))

        idx = np.arange(m)
        ring = np.minimum(np.abs(idx - bmu), m - np.abs(idx - bmu))
        influence = np.exp(-(ring ** 2) / (2 * self.radius * self.radius))
        self.neurons = self.neurons + (self.lr * influence)[:, None] * (cxy - self.neurons)

        self.iteration += 1
        self.bmu_index = bmu
        self.current_city = city_idx

        self.cost.distance_evals += m
        self.cost.unit_updates += m
        self.cost.stop()

        if self.iteration >= self.max_iterations:
            self._done_steps.add("train")

        return self._frame("train", "train",
                           f"iter {self.iteration}/{self.max_iterations} · city {city_idx} · BMU {bmu}",
                           highlight_nodes=[city_idx],
                           notes={"lr": f"{self.lr:.4f}", "radius": f"{self.radius:.3f}"})

    def _do_build_tour(self) -> VisualFrame:
        assert self.neurons is not None
        self.cost.start()
        assignments = []
        m = len(self.neurons)
        for city in self.nodes:
            diff = self.neurons - np.array([city.x, city.y])
            d2 = np.einsum("ij,ij->i", diff, diff)
            self.cost.distance_evals += m
            bmu = int(np.argmin(d2))
            assignments.append((bmu, float(d2[bmu]), city.idx))
        assignments.sort(key=lambda x: (x[0], x[1]))
        order, seen = [], set()
        for _, _, cid in assignments:
            if cid not in seen:
                seen.add(cid)
                order.append(cid)
        length = tour_length(self.nodes, order)
        self.tours["som"] = TourResult("SOM tour", order, length, kind="heuristic")
        self.cost.stop()
        self._done_steps.add("build_tour")
        return self._frame("build_tour", "tour", f"SOM tour built. length={length:.4f}",
                           notes={"length": f"{length:.4f}"})

    def _do_two_opt(self) -> VisualFrame:
        base = self.tours.get("som")
        if base is None:
            self._do_build_tour()
            base = self.tours["som"]
        self.cost.start()
        order, length, imp = two_opt(self.nodes, base.order,
                                     max_passes=int(self.params.get("two_opt_passes", 50)),
                                     meter=self.cost)
        self.cost.stop()
        self.tours["opt"] = TourResult("SOM + 2-opt", order, length, kind="refined",
                                       meta={"improvements": imp})
        self._done_steps.add("two_opt")
        return self._frame("two_opt", "two_opt",
                           f"2-opt done. length={length:.4f}, improvements={imp}",
                           notes={"length": f"{length:.4f}", "improvements": str(imp)})

    def _do_exact(self) -> VisualFrame:
        self.cost.start()
        order, length, msg = held_karp_exact(
            self.nodes, max_n=int(self.params.get("exact_max_n", 12)), meter=self.cost)
        self.cost.stop()
        self.exact_message = msg
        if order:
            self.tours["exact"] = TourResult("Exact optimum", order, length, kind="optimal")
        self._done_steps.add("exact")
        return self._frame("exact", "exact", msg,
                           notes=({"length": f"{length:.4f}"} if order else {}))

    # ------------------------------------------------------------------ #
    def micro_step(self) -> Optional[VisualFrame]:
        sid = self.active_step_id
        if sid is None:
            return None
        if self.status(sid) == "locked":
            return self._frame(sid, "blocked", f"'{self._step(sid).label}' locked.")
        if len(self.nodes) < 2 and sid in ("init", "train", "build_tour", "two_opt"):
            return self._frame(sid, "blocked", "Need at least 2 points for SOM.")

        frame = None
        if sid == "init":
            frame = self._do_init()
        elif sid == "train":
            if "init" not in self._done_steps:
                self._do_init()
            frame = self._train_one()
        elif sid == "build_tour":
            frame = self._do_build_tour()
        elif sid == "two_opt":
            frame = self._do_two_opt()
        elif sid == "exact":
            frame = self._do_exact()

        # advance cursor to next ready step when current one is finished
        if self.is_step_done(sid) or (sid == "exact"):
            nxt = self.next_ready_step()
            if nxt is not None and nxt != sid:
                self.active_step_id = nxt
        return frame

    # ------------------------------------------------------------------ #
    def result(self) -> dict:
        d = super().result()
        d.update({
            "neurons": self.neurons.tolist() if self.neurons is not None else [],
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "bmu_index": self.bmu_index,
            "exact_message": self.exact_message,
            "tours": {k: v.to_dict() for k, v in self.tours.items()},
        })
        return d

    def load_result(self, data: dict) -> None:
        super().load_result(data)
        ns = data.get("neurons", [])
        self.neurons = np.array(ns, dtype=float) if ns else None
        self.iteration = int(data.get("iteration", 0))
        self.max_iterations = int(data.get("max_iterations", self.max_iterations))
        self.bmu_index = data.get("bmu_index")
        self.exact_message = data.get("exact_message", "")
        self.tours = {k: TourResult.from_dict(v) for k, v in data.get("tours", {}).items()}

    def comparison_rows(self) -> List[dict]:
        rows = []
        for key, t in self.tours.items():
            rows.append({
                "algorithm": self.name, "variant": t.label, "kind": t.kind,
                "length": t.length, "n_nodes": len(t.order),
                "steps": self.cost.logical_steps, "dist_evals": self.cost.distance_evals,
                "ms": round(self.cost.elapsed_ms, 1),
            })
        return rows

    # ------------------------------------------------------------------ #
    def layers(self) -> List[LayerSpec]:
        return [
            LayerSpec("ring", "SOM ring", True),
            LayerSpec("som_tour", "SOM tour", True),
            LayerSpec("opt_tour", "2-opt tour", True),
            LayerSpec("exact_tour", "Exact tour", True),
        ]

    def draw(self, ax, ctx: DrawContext) -> None:
        by_id = ctx.by_id
        on = ctx.layer_on
        # exact (faint, thick, under everything)
        if on.get("exact_tour", True) and "exact" in self.tours:
            for a, b in self.tours["exact"].edges:
                if a in by_id and b in by_id:
                    ax.plot([by_id[a].x, by_id[b].x], [by_id[a].y, by_id[b].y],
                            color="#2ca02c", lw=3.2, alpha=0.22, zorder=ctx.z_base, solid_capstyle="round")
        # som tour
        if on.get("som_tour", True) and "som" in self.tours:
            for a, b in self.tours["som"].edges:
                if a in by_id and b in by_id:
                    ax.plot([by_id[a].x, by_id[b].x], [by_id[a].y, by_id[b].y],
                            color=self.color, lw=1.8, alpha=0.55, zorder=ctx.z_base + 1)
        # 2-opt tour
        if on.get("opt_tour", True) and "opt" in self.tours:
            for a, b in self.tours["opt"].edges:
                if a in by_id and b in by_id:
                    ax.plot([by_id[a].x, by_id[b].x], [by_id[a].y, by_id[b].y],
                            color="#0b3d91", lw=2.2, ls="--", alpha=0.8, zorder=ctx.z_base + 2)
        # ring + neurons + BMU
        if on.get("ring", True) and self.neurons is not None and len(self.neurons):
            xs = np.append(self.neurons[:, 0], self.neurons[0, 0])
            ys = np.append(self.neurons[:, 1], self.neurons[0, 1])
            ax.plot(xs, ys, color="#6baed6", lw=1.2, alpha=0.7,
                    label="SOM ring", zorder=ctx.z_base + 3)
            ax.scatter(self.neurons[:, 0], self.neurons[:, 1], s=6,
                       color="#6baed6", alpha=0.7, zorder=ctx.z_base + 4)
            if self.bmu_index is not None and 0 <= self.bmu_index < len(self.neurons):
                bx, by = self.neurons[self.bmu_index]
                ax.scatter([bx], [by], s=130, marker="x", color="#ff7f0e",
                           label="BMU", zorder=ctx.z_base + 5)
