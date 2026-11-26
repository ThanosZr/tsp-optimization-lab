"""Your custom algorithm.

Stage 1: recursive mutual-nearest pairing.
Stage 2: strict local plausibility rejection.
Stage 3: translucent local regions (areas of influence, NOT a final route).
Stages 4-7: perimeter construction, interior insertion, Hamiltonian
expansion and bounded 2-opt repair.

The Stage 1-3 logic is retained from the legacy v3 implementation and
wrapped on the Algorithm contract with cost instrumentation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Set, Tuple

from core.algorithm import (Algorithm, DrawContext, LayerSpec, Step, VisualFrame)
from core.models import Node, TourResult, tour_length
from core.registry import register_algorithm


@dataclass
class RegionComponent:
    cid: int
    node_ids: List[int]
    center_x: float
    center_y: float
    radius: float
    accepted_edge: Optional[Tuple[int, int]] = None
    source: str = "region"

    def to_dict(self) -> dict:
        return {"cid": self.cid, "node_ids": list(self.node_ids),
                "center_x": self.center_x, "center_y": self.center_y,
                "radius": self.radius, "accepted_edge": self.accepted_edge,
                "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "RegionComponent":
        ae = d.get("accepted_edge")
        return cls(d["cid"], list(d["node_ids"]), d["center_x"], d["center_y"],
                   d["radius"], tuple(ae) if ae else None, d.get("source", "region"))


@register_algorithm
class CustomRegionsAlgorithm(Algorithm):
    id = "custom"
    name = "Custom regions"
    color = "#d62728"      # red/warm family -> visually distinct from the blue SOM

    STEPS = [
        Step("stage1", "Stage 1 · Pairing", requires=[], produces=["stage1_pairs"],
             description="Recursive mutual-nearest pairing."),
        Step("stage2", "Stage 2 · Filter", requires=["stage1"], produces=["accepted", "rejected"],
             description="Strict local plausibility rejection."),
        Step("stage3", "Stage 3 · Regions", requires=["stage2"], produces=["regions"],
             description="Accepted pairs + leftovers become translucent regions."),
        Step("stage4", "Stage 4 · Perimeter", requires=["stage3"], produces=["perimeter"],
             description="Outline first: convex-hull order of the outer regions."),
        Step("stage5", "Stage 5 · Insert interior", requires=["stage4"], produces=["region_cycle"],
             description="Cheapest-insertion of interior regions into the outline."),
        Step("stage6", "Stage 6 · Expand", requires=["stage5"], produces=["custom_tour"],
             description="Expand the region cycle into one Hamiltonian node tour."),
        Step("stage7", "Stage 7 · Repair", requires=["stage6"], produces=["custom_tour"],
             description="Targeted long-edge repair; keep suspect regions in check.",
             rerunnable=True),
    ]

    def __init__(self, nodes: List[Node], params: Dict[str, object]) -> None:
        super().__init__(nodes, params)
        self.rng = random.Random(int(params.get("seed", 7)))
        self.generator: Optional[Generator[VisualFrame, None, None]] = None
        self._gen_for: Optional[str] = None
        self.stage1_pairs: List[Tuple[int, int]] = []
        self.accepted_pairs: List[Tuple[int, int]] = []
        self.rejected_pairs: List[Tuple[int, int]] = []
        self.candidate_edges: List[Tuple[int, int]] = []
        self.available_nodes: Set[int] = set()
        self.regions: List[RegionComponent] = []
        self.current_node: Optional[int] = None
        self.current_pair: Optional[Tuple[int, int]] = None
        self.tour: Optional[TourResult] = None
        self.region_cycle: List[int] = []          # region indices in tour order
        self.perimeter: List[int] = []             # region indices on the outline (hull)
        self.suspect_regions: List[int] = []       # large-radius regions kept under review

    # ------------------------------------------------------------------ #
    def reset(self) -> VisualFrame:
        super().reset()
        self.rng = random.Random(int(self.params.get("seed", 7)))
        self.generator = None
        self._gen_for = None
        self.stage1_pairs = []
        self.accepted_pairs = []
        self.rejected_pairs = []
        self.candidate_edges = []
        self.available_nodes = {n.idx for n in self.nodes}
        self.regions = []
        self.current_node = None
        self.current_pair = None
        self.tour = None
        self.region_cycle = []
        self.perimeter = []
        self.suspect_regions = []
        return self._frame("stage1", "reset", "Custom reset.")

    # ------------------------------------------------------------------ #
    def micro_step(self) -> Optional[VisualFrame]:
        sid = self.active_step_id
        if sid is None:
            return None
        if self.status(sid) == "locked":
            return self._frame(sid, "blocked",
                               f"'{self._step(sid).label}' locked; run prerequisites.")
        if len(self.nodes) < 2:
            return self._frame(sid, "blocked", "Need at least 2 points.")

        self.cost.start()
        if self.generator is None or self._gen_for != sid:
            self.generator = {
                "stage1": self._stage1_pairing,
                "stage2": self._stage2_filtering,
                "stage3": self._stage3_regions,
                "stage4": self._stage4_perimeter,
                "stage5": self._stage5_insert_interior,
                "stage6": self._stage6_expand_to_tour,
                "stage7": self._stage7_repair,
            }[sid]()
            self._gen_for = sid
        try:
            frame = next(self.generator)
            self.cost.stop()
            return frame
        except StopIteration:
            self.cost.stop()
            self.generator = None
            self._gen_for = None
            self._done_steps.add(sid)
            nxt = self.next_ready_step()
            if nxt is not None and nxt != sid:
                self.active_step_id = nxt
            return self._frame(sid, "phase", f"{self._step(sid).label} complete.")

    # ---- Stage 1 (verbatim logic, instrumented) ----------------------- #
    def _stage1_pairing(self):
        nodes = self.nodes
        self.stage1_pairs = []
        self.accepted_pairs = []
        self.rejected_pairs = []
        self.regions = []
        self.candidate_edges = []
        available: Set[int] = {n.idx for n in nodes}
        self.available_nodes = set(available)
        unresolved_stack: List[int] = []
        min_candidates = int(self.params.get("custom_min_candidates", 8))
        exact_search = bool(self.params.get("custom_exact_search", False))
        yield self._frame("stage1", "stage1", "Stage 1: recursive mutual-nearest pairing.",
                          notes={"available": str(len(available))})
        while len(available) >= 2:
            while unresolved_stack and unresolved_stack[-1] not in available:
                unresolved_stack.pop()
            if unresolved_stack:
                start = unresolved_stack.pop()
                msg = f"Restart from unresolved node {start}."
            else:
                start = self.rng.choice(sorted(available))
                msg = f"Start chain from node {start}."
            chain = [start]
            self.current_node = start
            self.current_pair = None
            self.candidate_edges = []
            yield self._frame("stage1", "stage1", msg, highlight_nodes=[start])
            while True:
                current = chain[-1]
                pool = set(available)
                pool.discard(current)
                nearest = self._nearest_node(current, pool, min_candidates=min_candidates, exact=exact_search)
                if nearest is None:
                    yield self._frame("stage1", "stage1", f"Node {current} has no candidate.",
                                      highlight_nodes=[current])
                    break
                nxt, dist = nearest
                self.current_node = current
                self.current_pair = (current, nxt)
                self.candidate_edges.append((current, nxt))
                yield self._frame("stage1", "stage1", f"Node {current} -> candidate {nxt} (d={dist:.4f}).",
                                  highlight_nodes=[current], highlight_edges=[(current, nxt)],
                                  notes={"candidate": f"{current}-{nxt}", "distance": f"{dist:.4f}"})
                if len(chain) >= 2 and nxt == chain[-2]:
                    a, b = chain[-2], current
                    self.stage1_pairs.append((a, b))
                    available.remove(a); available.remove(b)
                    self.available_nodes = set(available)
                    losers = [cid for cid in chain[:-2] if cid in available]
                    unresolved_stack.extend(reversed(losers))
                    self.candidate_edges = []
                    self.current_pair = (a, b)
                    yield self._frame("stage1", "stage1",
                                      f"Lock mutual pair {a}-{b}. Unresolved: {losers if losers else 'none'}.",
                                      highlight_edges=[(a, b)],
                                      notes={"locked_pair": f"{a}-{b}", "pairs": str(len(self.stage1_pairs))})
                    break
                if nxt in chain:
                    cycle = chain[chain.index(nxt):] + [nxt]
                    cycle_edges = list(zip(cycle[:-1], cycle[1:]))
                    a, b = min(cycle_edges, key=lambda e: self._node_distance(e[0], e[1]))
                    if a in available and b in available:
                        self.stage1_pairs.append((a, b))
                        available.remove(a); available.remove(b)
                        self.available_nodes = set(available)
                    losers = [u for u in chain if u in available]
                    unresolved_stack.extend(reversed(losers))
                    self.candidate_edges = []
                    self.current_pair = (a, b)
                    yield self._frame("stage1", "stage1",
                                      f"Cycle {cycle}. Locked shortest edge {a}-{b}; unresolved {losers}.",
                                      highlight_edges=[(a, b)],
                                      notes={"locked_pair": f"{a}-{b}", "pairs": str(len(self.stage1_pairs))})
                    break
                chain.append(nxt)
        if len(available) == 1:
            last = next(iter(available))
            yield self._frame("stage1", "stage1", f"Odd leftover node {last} -> singleton region later.",
                              highlight_nodes=[last])
        self.candidate_edges = []
        self.available_nodes = set(available)
        yield self._frame("stage1", "stage1",
                          f"Stage 1 complete. pairs={len(self.stage1_pairs)}, leftovers={len(available)}.",
                          notes={"pairs": str(len(self.stage1_pairs)), "leftovers": str(len(available))})

    # ---- Stage 2 ------------------------------------------------------- #
    def _stage2_filtering(self):
        self.accepted_pairs = []
        self.rejected_pairs = []
        k = int(self.params.get("custom_filter_k", 8))
        rank_limit = int(self.params.get("custom_max_rank", 3))
        best_ratio_limit = float(self.params.get("custom_max_best_ratio", 2.0))
        median_ratio_limit = float(self.params.get("custom_max_median_ratio", 1.7))
        one_side = bool(self.params.get("custom_reject_one_side", True))
        yield self._frame("stage2", "stage2", f"Stage 2: reviewing {len(self.stage1_pairs)} pairs.")
        for a, b in self.stage1_pairs:
            ra, besta, meda, da = self._local_stats(a, b, k)
            rb, bestb, medb, db = self._local_stats(b, a, k)
            eps = 1e-12
            a_best_ratio, b_best_ratio = da / max(besta, eps), db / max(bestb, eps)
            a_med_ratio, b_med_ratio = da / max(meda, eps), db / max(medb, eps)
            bad_a = ra > rank_limit or a_best_ratio > best_ratio_limit or a_med_ratio > median_ratio_limit
            bad_b = rb > rank_limit or b_best_ratio > best_ratio_limit or b_med_ratio > median_ratio_limit
            reject = (bad_a or bad_b) if one_side else (bad_a and bad_b)
            self.current_pair = (a, b)
            notes = {"rank_a": str(ra), "rank_b": str(rb),
                     "a_best_ratio": f"{a_best_ratio:.2f}", "b_best_ratio": f"{b_best_ratio:.2f}",
                     "a_median_ratio": f"{a_med_ratio:.2f}", "b_median_ratio": f"{b_med_ratio:.2f}"}
            if reject:
                self.rejected_pairs.append((a, b)); msg = f"REJECT {a}-{b} (bad_a={bad_a}, bad_b={bad_b})."
            else:
                self.accepted_pairs.append((a, b)); msg = f"ACCEPT {a}-{b}."
            yield self._frame("stage2", "stage2", msg, highlight_edges=[(a, b)], notes=notes)
        yield self._frame("stage2", "stage2",
                          f"Stage 2 complete. accepted={len(self.accepted_pairs)}, rejected={len(self.rejected_pairs)}.",
                          notes={"accepted": str(len(self.accepted_pairs)), "rejected": str(len(self.rejected_pairs))})

    # ---- Stage 3 ------------------------------------------------------- #
    def _stage3_regions(self):
        self.regions = []
        used: Set[int] = set()
        cid = 0
        region_padding = float(self.params.get("custom_region_padding", 0.035))
        singleton_radius = float(self.params.get("custom_singleton_radius", 0.045))
        yield self._frame("stage3", "stage3", "Stage 3: building translucent regions.")
        for a, b in self.accepted_pairs:
            pa, pb = self._node(a), self._node(b)
            cx, cy = (pa.x + pb.x) / 2, (pa.y + pb.y) / 2
            r = max(self._node_distance(a, b) / 2 + region_padding, singleton_radius)
            self.regions.append(RegionComponent(cid, [a, b], cx, cy, r, (a, b), "accepted_pair_region"))
            used.update([a, b])
            self.current_pair = (a, b)
            yield self._frame("stage3", "stage3", f"Region R{cid} from pair {a}-{b} (r={r:.4f}).",
                              highlight_edges=[(a, b)],
                              notes={"region": f"R{cid}", "center": f"({cx:.3f},{cy:.3f})", "radius": f"{r:.4f}"})
            cid += 1
        for node_id in sorted({n.idx for n in self.nodes} - used):
            p = self._node(node_id)
            self.regions.append(RegionComponent(cid, [node_id], p.x, p.y, singleton_radius, None,
                                                "singleton_or_rejected_region"))
            self.current_node = node_id
            yield self._frame("stage3", "stage3", f"Singleton region R{cid} from node {node_id}.",
                              highlight_nodes=[node_id], notes={"region": f"R{cid}"})
            cid += 1
        yield self._frame("stage3", "stage3",
                          f"Stage 3 complete. regions={len(self.regions)}. No final route locked.",
                          notes={"regions": str(len(self.regions))})

    # ---- Stage 4: PERIMETER (outline first) --------------------------- #
    def _region_centers(self) -> List[Tuple[float, float]]:
        return [(r.center_x, r.center_y) for r in self.regions]

    def _convex_hull_indices(self, pts: List[Tuple[float, float]]) -> List[int]:
        """Andrew's monotone chain hull -> region indices in CCW order."""
        order = sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
        self.cost.distance_evals += len(pts)
        if len(order) <= 2:
            return order

        def cross(o, a, b):
            return ((pts[a][0] - pts[o][0]) * (pts[b][1] - pts[o][1])
                    - (pts[a][1] - pts[o][1]) * (pts[b][0] - pts[o][0]))

        lower: List[int] = []
        for i in order:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
                lower.pop()
            lower.append(i)
        upper: List[int] = []
        for i in reversed(order):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
                upper.pop()
            upper.append(i)
        return lower[:-1] + upper[:-1]

    def _stage4_perimeter(self):
        """Build the outline: the convex hull of region centres defines the
        outer ring every interior region/node will later attach to."""
        self.region_cycle = []
        self.perimeter = []
        # mark suspect (large-radius) regions; they stay under review to the end
        if self.regions:
            radii = sorted(r.radius for r in self.regions)
            med = radii[len(radii) // 2]
            thr = float(self.params.get("custom_suspect_radius_factor", 2.2)) * max(med, 1e-9)
            self.suspect_regions = [i for i, r in enumerate(self.regions) if r.radius >= thr]
        yield self._frame("stage4", "stage4",
                          f"Outline: hull of {len(self.regions)} region centres. "
                          f"suspect(large) regions: {self.suspect_regions or 'none'}.",
                          notes={"regions": str(len(self.regions)),
                                 "suspect": str(len(self.suspect_regions))})
        if not self.regions:
            return
        pts = self._region_centers()
        self.perimeter = self._convex_hull_indices(pts)
        self.region_cycle = list(self.perimeter)
        for ri in self.perimeter:
            self.current_node = self.regions[ri].node_ids[0]
            yield self._frame("stage4", "stage4", f"Perimeter region R{ri} on the outline.",
                              highlight_nodes=list(self.regions[ri].node_ids))
        yield self._frame("stage4", "stage4",
                          f"Outline built with {len(self.perimeter)} regions.",
                          notes={"perimeter": str(len(self.perimeter))})

    # ---- Stage 5: INSERT INTERIOR (cheapest insertion) ---------------- #
    def _stage5_insert_interior(self):
        """Gradually attach the remaining (interior) regions to the outline by
        cheapest insertion. Insertion order does not matter for correctness."""
        if not self.region_cycle:
            self.region_cycle = list(range(len(self.regions)))
            yield self._frame("stage5", "stage5", "No outline; using natural region order.")
            return
        pts = self._region_centers()

        def d(i, j):
            self.cost.distance_evals += 1
            return math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])

        remaining = [i for i in range(len(self.regions)) if i not in set(self.region_cycle)]
        # insert nearest-to-outline first (stable, cheap); order is not critical
        remaining.sort(key=lambda r: min((d(r, c) for c in self.region_cycle), default=0.0))
        yield self._frame("stage5", "stage5",
                          f"Inserting {len(remaining)} interior regions into the outline.")
        for r in remaining:
            best_pos, best_delta = 0, None
            m = len(self.region_cycle)
            for pos in range(m):
                i = self.region_cycle[pos]
                j = self.region_cycle[(pos + 1) % m]
                delta = d(i, r) + d(r, j) - d(i, j)
                if best_delta is None or delta < best_delta:
                    best_delta, best_pos = delta, pos
            self.region_cycle.insert(best_pos + 1, r)
            self.current_node = self.regions[r].node_ids[0]
            yield self._frame("stage5", "stage5",
                              f"Insert R{r} after position {best_pos} (Δ={best_delta:.4f}).",
                              highlight_nodes=list(self.regions[r].node_ids),
                              notes={"region": f"R{r}", "delta": f"{best_delta:.4f}"})
        yield self._frame("stage5", "stage5",
                          f"Region cycle complete: {len(self.region_cycle)} regions.",
                          notes={"region_cycle": str(len(self.region_cycle))})

    # ---- Stage 6: EXPAND region cycle -> node tour -------------------- #
    def _stage6_expand_to_tour(self):
        """Turn the region cycle into one Hamiltonian node tour. Two-node
        (accepted-pair) regions are oriented to chain smoothly; every node
        appears exactly once, so the result is a single closed loop."""
        if not self.region_cycle:
            yield self._frame("stage6", "stage6", "No region cycle; nothing to expand.")
            return

        def dist(a, b):
            return self._node_distance(a, b)

        order: List[int] = []
        tail: Optional[int] = None
        for ri in self.region_cycle:
            members = list(self.regions[ri].node_ids)
            if len(members) == 1:
                order.append(members[0]); tail = members[0]
            else:
                a, b = members[0], members[1]
                if tail is None or dist(tail, a) <= dist(tail, b):
                    order.extend([a, b]); tail = b
                else:
                    order.extend([b, a]); tail = a
        # de-duplicate defensively (regions partition nodes, but be safe)
        seen, clean = set(), []
        for nid in order:
            if nid not in seen:
                seen.add(nid); clean.append(nid)
        for n in self.nodes:                       # ensure completeness
            if n.idx not in seen:
                clean.append(n.idx); seen.add(n.idx)
        length = tour_length(self.nodes, clean)
        self.tour = TourResult("Custom route", clean, length, kind="heuristic",
                               meta={"stage": "expanded"})
        yield self._frame("stage6", "stage6",
                          f"Expanded to a single tour over {len(clean)} nodes. length={length:.4f}.",
                          notes={"length": f"{length:.4f}", "nodes": str(len(clean))})

    # ---- Stage 7: REPAIR (targeted long-edge 2-opt) ------------------- #
    def _stage7_repair(self):
        """Cheap repair: remove the worst (longest) edges with bounded 2-opt.
        This is where suspect large regions get their crossings fixed without a
        full optimisation pass."""
        if not self.tour or len(self.tour.order) < 4:
            yield self._frame("stage7", "stage7", "Tour too small to repair.")
            return
        from algorithms.tsp_utils import two_opt
        before = self.tour.length
        passes = int(self.params.get("custom_repair_passes", 6))
        yield self._frame("stage7", "stage7",
                          f"Repairing longest edges (≤{passes} passes). before={before:.4f}.")
        order, length, imp = two_opt(self.nodes, self.tour.order, max_passes=passes, meter=self.cost)
        self.tour = TourResult("Custom route", order, length, kind="heuristic",
                               meta={"stage": "repaired", "improvements": imp})
        yield self._frame("stage7", "stage7",
                          f"Repair done. before={before:.4f} after={length:.4f} "
                          f"(improvements={imp}).",
                          notes={"length": f"{length:.4f}", "improvements": str(imp)})

    # ---- geometry helpers (instrumented) ------------------------------ #
    def _node(self, idx: int) -> Node:
        return self.node_by_id()[idx]

    def _node_distance(self, a: int, b: int) -> float:
        self.cost.distance_evals += 1
        pa, pb = self._node(a), self._node(b)
        return math.hypot(pa.x - pb.x, pa.y - pb.y)

    def _nearest_node(self, item_id: int, pool: Set[int], *, min_candidates: int, exact: bool):
        if not pool:
            return None
        if exact or len(pool) <= min_candidates:
            best = min(pool, key=lambda j: self._node_distance(item_id, j))
            return best, self._node_distance(item_id, best)
        by_id = self.node_by_id()
        p = by_id[item_id]
        sx = sorted(pool, key=lambda j: abs(by_id[j].x - p.x))
        sy = sorted(pool, key=lambda j: abs(by_id[j].y - p.y))
        pr, pphi = math.hypot(p.x, p.y), math.atan2(p.y, p.x)
        sr = sorted(pool, key=lambda j: abs(math.hypot(by_id[j].x, by_id[j].y) - pr))
        sphi = sorted(pool, key=lambda j: abs(math.atan2(by_id[j].y, by_id[j].x) - pphi))
        cand = set(sx[:min_candidates] + sy[:min_candidates] + sr[:min_candidates] + sphi[:min_candidates]) or set(pool)
        best = min(cand, key=lambda j: self._node_distance(item_id, j))
        return best, self._node_distance(item_id, best)

    def _local_stats(self, a: int, b: int, k: int):
        by_id = self.node_by_id()
        pa = by_id[a]
        distances = []
        for n in self.nodes:
            if n.idx == a:
                continue
            self.cost.distance_evals += 1
            distances.append((n.idx, math.hypot(pa.x - n.x, pa.y - n.y)))
        distances.sort(key=lambda x: x[1])
        edge_d = self._node_distance(a, b)
        best_d = distances[0][1] if distances else edge_d
        top = distances[: max(1, min(k, len(distances)))]
        median_d = sorted(d for _, d in top)[len(top) // 2] if top else edge_d
        rank = len(distances) + 1
        for i, (j, _) in enumerate(distances, start=1):
            if j == b:
                rank = i
                break
        return rank, best_d, median_d, edge_d

    # ---- results / serialisation -------------------------------------- #
    def result(self) -> dict:
        d = super().result()
        d.update({
            "stage1_pairs": [list(p) for p in self.stage1_pairs],
            "accepted_pairs": [list(p) for p in self.accepted_pairs],
            "rejected_pairs": [list(p) for p in self.rejected_pairs],
            "regions": [r.to_dict() for r in self.regions],
            "region_cycle": list(self.region_cycle),
            "perimeter": list(self.perimeter),
            "suspect_regions": list(self.suspect_regions),
            "tour": self.tour.to_dict() if self.tour else None,
        })
        return d

    def load_result(self, data: dict) -> None:
        super().load_result(data)
        self.stage1_pairs = [tuple(p) for p in data.get("stage1_pairs", [])]
        self.accepted_pairs = [tuple(p) for p in data.get("accepted_pairs", [])]
        self.rejected_pairs = [tuple(p) for p in data.get("rejected_pairs", [])]
        self.regions = [RegionComponent.from_dict(r) for r in data.get("regions", [])]
        self.region_cycle = list(data.get("region_cycle", []))
        self.perimeter = list(data.get("perimeter", []))
        self.suspect_regions = list(data.get("suspect_regions", []))
        t = data.get("tour")
        self.tour = TourResult.from_dict(t) if t else None

    def comparison_rows(self) -> List[dict]:
        rows = [{
            "algorithm": self.name, "variant": "Stage 1-3 (regions)", "kind": "structure",
            "length": None, "n_nodes": len(self.nodes),
            "steps": self.cost.logical_steps, "dist_evals": self.cost.distance_evals,
            "ms": round(self.cost.elapsed_ms, 1),
            "extra": f"pairs={len(self.stage1_pairs)} acc={len(self.accepted_pairs)} "
                     f"rej={len(self.rejected_pairs)} reg={len(self.regions)}",
        }]
        if self.tour:
            rows.append({
                "algorithm": self.name, "variant": self.tour.label, "kind": self.tour.kind,
                "length": self.tour.length, "n_nodes": len(self.tour.order),
                "steps": self.cost.logical_steps, "dist_evals": self.cost.distance_evals,
                "ms": round(self.cost.elapsed_ms, 1), "extra": "",
            })
        return rows

    # ---- visual layers ------------------------------------------------- #
    def layers(self) -> List[LayerSpec]:
        return [
            LayerSpec("regions", "Regions", True),
            LayerSpec("perimeter", "Perimeter/outline", True),
            LayerSpec("accepted", "Accepted edges", True),
            LayerSpec("candidates", "Candidate edges", True),
            LayerSpec("rejected", "Rejected edges", True),
            LayerSpec("route", "Custom route", True),
        ]

    def draw(self, ax, ctx: DrawContext) -> None:
        from matplotlib.patches import Circle
        by_id = ctx.by_id
        on = ctx.layer_on
        palette = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
                   "#bcbd22", "#17becf", "#7f7f7f", "#2ca02c", "#1f77b4"]
        alpha = float(self.params.get("custom_region_alpha", 0.22))
        if on.get("regions", True):
            for region in self.regions:
                ax.add_patch(Circle((region.center_x, region.center_y), region.radius,
                                    facecolor=palette[region.cid % len(palette)], edgecolor="none",
                                    alpha=alpha, zorder=ctx.z_base))
                ax.text(region.center_x, region.center_y, f"R{region.cid}", fontsize=8,
                        alpha=0.75, ha="center", va="center", zorder=ctx.z_base + 1)
            # suspect (large-radius) regions get a dashed outline so they stand out
            for ri in self.suspect_regions:
                if 0 <= ri < len(self.regions):
                    r = self.regions[ri]
                    ax.add_patch(Circle((r.center_x, r.center_y), r.radius, fill=False,
                                        edgecolor="#7b241c", ls="--", lw=1.4,
                                        alpha=0.7, zorder=ctx.z_base + 1))
        if on.get("perimeter", True) and self.perimeter:
            pp = [self.regions[i] for i in self.perimeter if 0 <= i < len(self.regions)]
            if len(pp) >= 2:
                xs = [r.center_x for r in pp] + [pp[0].center_x]
                ys = [r.center_y for r in pp] + [pp[0].center_y]
                ax.plot(xs, ys, color="#1b5e20", lw=1.6, ls="-.", alpha=0.65,
                        label="outline", zorder=ctx.z_base + 5)
        if on.get("route", True) and self.tour:
            for a, b in self.tour.edges:
                if a in by_id and b in by_id:
                    ax.plot([by_id[a].x, by_id[b].x], [by_id[a].y, by_id[b].y],
                            color="#7b241c", lw=2.4, alpha=0.85, zorder=ctx.z_base + 6)
        if on.get("candidates", True):
            for a, b in self.candidate_edges:
                if a in by_id and b in by_id:
                    ax.plot([by_id[a].x, by_id[b].x], [by_id[a].y, by_id[b].y],
                            color="#ff7f0e", ls="--", lw=1.3, alpha=0.7, zorder=ctx.z_base + 2)
        if on.get("accepted", True):
            edges = self.accepted_pairs if self.accepted_pairs else self.stage1_pairs
            for a, b in edges:
                if a in by_id and b in by_id:
                    ax.plot([by_id[a].x, by_id[b].x], [by_id[a].y, by_id[b].y],
                            color=self.color, lw=2.0, alpha=0.8, zorder=ctx.z_base + 3)
        if on.get("rejected", True):
            for a, b in self.rejected_pairs:
                if a in by_id and b in by_id:
                    ax.plot([by_id[a].x, by_id[b].x], [by_id[a].y, by_id[b].y],
                            color="#7f7f7f", ls=":", lw=2.2, alpha=0.8, zorder=ctx.z_base + 4)
        if self.current_pair:
            a, b = self.current_pair
            if a in by_id and b in by_id:
                ax.scatter([by_id[a].x, by_id[b].x], [by_id[a].y, by_id[b].y],
                           s=110, marker="D", facecolors="none", edgecolors="#7b241c",
                           linewidths=2.0, zorder=ctx.z_base + 7)
