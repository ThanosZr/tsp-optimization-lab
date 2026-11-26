"""Headless / batch runner. Reuses the exact same core as the GUI.

Examples
--------
  python cli.py run --n 12 --seed 7 --algos som custom --export results.csv
  python cli.py benchmark --algo custom --n 30 --seeds 0-20
  python cli.py run --n 10 --algos som --png map.png
"""
from __future__ import annotations

import argparse
import statistics
import sys
from typing import List

import bootstrap  # noqa: F401  (populates registries)
from core.data import make_nodes
from core.registry import algorithm_classes, exporters
from core.session import Session


def _all_algo_ids() -> List[str]:
    return [c.id for c in algorithm_classes()]


def _parse_seeds(spec: str) -> List[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def cmd_run(args) -> None:
    nodes = make_nodes(args.n, seed=args.seed, mode=args.data)
    session = Session(nodes, seed=args.seed, data_mode=args.data)
    algos = args.algos or _all_algo_ids()
    for aid in algos:
        algo = session.algorithms[aid]
        algo.run_all()
        session.log(f"Ran {algo.name}.")
    print(f"\nInstance: n={args.n} seed={args.seed} data={args.data}\n")
    _print_table(session.comparison_rows())
    if args.export:
        exporters()["Results table (CSV)"]["fn"](session, args.export)
        print(f"\nWrote results -> {args.export}")
    if args.png:
        exporters()["Map (PNG)"]["fn"](session, args.png)
        print(f"Wrote map -> {args.png}")
    if args.save:
        session.save(args.save)
        print(f"Saved session -> {args.save}")


def cmd_benchmark(args) -> None:
    seeds = _parse_seeds(args.seeds)
    lengths, gaps, ms = [], [], []
    for sd in seeds:
        nodes = make_nodes(args.n, seed=sd, mode=args.data)
        session = Session(nodes, seed=sd, data_mode=args.data)
        session.algorithms[args.algo].run_all()
        if args.algo != "som":  # need an optimal/ref -> also run som's exact for gap
            session.algorithms["som"].run_step("exact")
        for r in session.comparison_rows():
            if r["algorithm"].lower().startswith(args.algo[:3]) and r.get("length"):
                lengths.append(r["length"])
                if r.get("gap_pct") is not None:
                    gaps.append(r["gap_pct"])
                ms.append(r["ms"])
                break
    print(f"\nBenchmark '{args.algo}' over {len(seeds)} seeds (n={args.n}, {args.data}):")
    if lengths:
        print(f"  length: mean={statistics.mean(lengths):.4f} "
              f"min={min(lengths):.4f} max={max(lengths):.4f}")
    if gaps:
        print(f"  gap%   : mean={statistics.mean(gaps):.2f} "
              f"min={min(gaps):.2f} max={max(gaps):.2f}")
    if ms:
        print(f"  ms    : mean={statistics.mean(ms):.1f}")


def _print_table(rows) -> None:
    cols = [("algorithm", 16), ("variant", 22), ("length", 9),
            ("gap_pct", 8), ("steps", 8), ("dist_evals", 12), ("ms", 8)]
    print("  ".join(h.ljust(w) for h, w in cols))
    print("-" * (sum(w for _, w in cols) + 2 * len(cols)))
    for r in rows:
        cells = []
        for h, w in cols:
            v = r.get(h)
            if v is None:
                s = "—"
            elif isinstance(v, float):
                s = f"{v:.3f}" if h != "gap_pct" else f"{v:.2f}"
            else:
                s = str(v)
            cells.append(s.ljust(w))
        print("  ".join(cells))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="TSP Workbench headless runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run algorithms on one instance")
    pr.add_argument("--n", type=int, default=12)
    pr.add_argument("--seed", type=int, default=7)
    pr.add_argument("--data", default="uniform")
    pr.add_argument("--algos", nargs="*", default=None)
    pr.add_argument("--export", default=None, help="results CSV path")
    pr.add_argument("--png", default=None, help="map PNG path")
    pr.add_argument("--save", default=None, help="session JSON path")
    pr.set_defaults(func=cmd_run)

    pb = sub.add_parser("benchmark", help="run one algorithm over many seeds")
    pb.add_argument("--algo", default="custom")
    pb.add_argument("--n", type=int, default=30)
    pb.add_argument("--data", default="uniform")
    pb.add_argument("--seeds", default="0-9", help="'0-20' or '1,2,3'")
    pb.set_defaults(func=cmd_benchmark)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
