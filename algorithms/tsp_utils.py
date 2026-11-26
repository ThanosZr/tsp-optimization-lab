"""Reusable TSP primitives: 2-opt local search and exact Held-Karp.

Both accept an optional `meter` (core.algorithm.CostMeter) so the workbench can
report real distance-evaluation counts instead of opaque "steps".
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

from core.models import Node, tour_length


def two_opt(nodes: List[Node], order: List[int], max_passes: int = 50,
            meter=None) -> Tuple[List[int], float, int]:
    if len(order) < 4:
        return list(order), tour_length(nodes, order), 0

    by_id = {n.idx: n for n in nodes}

    def d(a, b):
        if meter is not None:
            meter.distance_evals += 1
        pa, pb = by_id[a], by_id[b]
        return math.hypot(pa.x - pb.x, pa.y - pb.y)

    route = list(order)
    n = len(route)
    improvements = 0

    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = route[i]
            b = route[(i + 1) % n]
            for k in range(i + 2, n):
                if k == n - 1 and i == 0:
                    continue
                c = route[k]
                e = route[(k + 1) % n]
                if d(a, c) + d(b, e) + 1e-12 < d(a, b) + d(c, e):
                    route[i + 1:k + 1] = reversed(route[i + 1:k + 1])
                    improvements += 1
                    improved = True
        if not improved:
            break

    return route, tour_length(nodes, route), improvements


def held_karp_exact(nodes: List[Node], max_n: int = 12,
                    meter=None) -> Tuple[List[int], float, str]:
    n = len(nodes)
    if n == 0:
        return [], 0.0, "No nodes."
    if n > max_n:
        return [], 0.0, f"Exact disabled for n={n}; max_n={max_n}."

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y)
            if meter is not None:
                meter.distance_evals += 1

    dp: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for j in range(1, n):
        dp[((1 << 0) | (1 << j), j)] = (dist[0][j], 0)

    for size in range(3, n + 1):
        for subset in itertools.combinations(range(1, n), size - 1):
            mask = 1
            for x in subset:
                mask |= 1 << x
            for j in subset:
                prev_mask = mask & ~(1 << j)
                best = None
                for k in subset:
                    if k == j or (prev_mask, k) not in dp:
                        continue
                    cost = dp[(prev_mask, k)][0] + dist[k][j]
                    if best is None or cost < best[0]:
                        best = (cost, k)
                if best is not None:
                    dp[(mask, j)] = best

    full = (1 << n) - 1
    best_end = None
    for j in range(1, n):
        if (full, j) not in dp:
            continue
        cost = dp[(full, j)][0] + dist[j][0]
        if best_end is None or cost < best_end[0]:
            best_end = (cost, j)
    if best_end is None:
        return [], 0.0, "Exact failed."

    best_cost, end = best_end
    order_rev = [end]
    mask, j = full, end
    while j != 0:
        _, prev = dp[(mask, j)]
        mask &= ~(1 << j)
        j = prev
        if j != 0:
            order_rev.append(j)
    order = [0] + list(reversed(order_rev))
    return order, best_cost, f"Exact optimal computed for n={n}."
