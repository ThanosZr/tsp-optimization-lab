"""Exporters. Each is registered so it appears in the GUI export menu and the
CLI automatically. Add a format = one decorated function."""
from __future__ import annotations

import csv
import json

from core.registry import register_exporter
from core.render import render


@register_exporter("Nodes (CSV)", "csv")
def export_nodes_csv(session, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "x", "y"])
        for n in session.nodes:
            w.writerow([n.idx, f"{n.x:.6f}", f"{n.y:.6f}"])


@register_exporter("Results table (CSV)", "csv")
def export_results_csv(session, path: str) -> None:
    rows = session.comparison_rows()
    cols = ["algorithm", "variant", "kind", "length", "gap_pct",
            "n_nodes", "steps", "dist_evals", "ms", "extra"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


@register_exporter("Tours (CSV)", "csv")
def export_tours_csv(session, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "variant", "length", "order"])
        for algo in session.algos():
            res = algo.result()
            tours = res.get("tours", {})
            if isinstance(tours, dict):
                for t in tours.values():
                    w.writerow([algo.name, t.get("label", ""), t.get("length", ""),
                                " ".join(map(str, t.get("order", [])))])
            single = res.get("tour")
            if single:
                w.writerow([algo.name, single.get("label", ""), single.get("length", ""),
                            " ".join(map(str, single.get("order", [])))])


@register_exporter("Map (PNG)", "png")
def export_map_png(session, path: str, layer_on=None, title="TSP Workbench") -> None:
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8.5, 8.5), dpi=130)
    ax = fig.add_subplot(111)
    render(ax, session.nodes, session.algos(), layer_on=layer_on, title=title)
    fig.savefig(path, bbox_inches="tight")


@register_exporter("Session (JSON)", "json")
def export_session_json(session, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, indent=2)
