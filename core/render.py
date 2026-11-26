"""Render a set of algorithms + node cloud onto a matplotlib Axes.

Used by the GUI (live canvas), the PNG exporter and the CLI -- one drawing path,
no duplication. It only depends on matplotlib + the Algorithm.draw contract.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from core.algorithm import Algorithm, DrawContext
from core.models import Node


def render(ax, nodes: List[Node], algorithms: List[Algorithm],
           layer_on: Optional[Dict[str, bool]] = None,
           emphasis_algo: Optional[str] = None,
           selected: Optional[List[int]] = None,
           current_highlight: Optional[List[int]] = None,
           show_labels: bool = True, title: str = "TSP Workbench") -> None:
    layer_on = layer_on or {}
    selected = selected or []
    current_highlight = current_highlight or []
    by_id = {n.idx: n for n in nodes}

    ax.clear()
    ax.set_title(title)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)

    # algorithm overlays, stacked by registration order
    z = 1
    for algo in algorithms:
        sub = {ls.id: layer_on.get(f"{algo.id}:{ls.id}", ls.default_on) for ls in algo.layers()}
        ctx = DrawContext(by_id=by_id, layer_on=sub,
                          emphasis=(emphasis_algo == algo.id), z_base=z)
        algo.draw(ax, ctx)
        z += 10

    # node cloud on top
    if nodes:
        ax.scatter([p.x for p in nodes], [p.y for p in nodes], s=42,
                   color="#2ca02c", alpha=0.9, label="cities", zorder=z + 10)
        if show_labels and len(nodes) <= 300:
            for p in nodes:
                ax.text(p.x + 0.006, p.y + 0.006, str(p.idx), fontsize=8, zorder=z + 11)

    sel = [by_id[i] for i in selected if i in by_id]
    if sel:
        ax.scatter([p.x for p in sel], [p.y for p in sel], s=180, marker="o",
                   facecolors="none", edgecolors="#000000", linewidths=2.0,
                   label="selected", zorder=z + 12)
    cur = [by_id[i] for i in current_highlight if i in by_id]
    if cur:
        ax.scatter([p.x for p in cur], [p.y for p in cur], s=150, marker="s",
                   facecolors="none", edgecolors="#ff7f0e", linewidths=2.4,
                   label="current", zorder=z + 13)

    handles, labels = ax.get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for h, l in zip(handles, labels):
        if l.startswith("_") or l in seen:
            continue
        seen.add(l); hh.append(h); ll.append(l)
    if hh:
        ax.legend(hh, ll, loc="upper right", fontsize=8, framealpha=0.85)
