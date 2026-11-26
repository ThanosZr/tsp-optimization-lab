# TSP Optimization Lab — Workbench 0.1.0

An interactive desktop workbench for the Euclidean Travelling Salesman Problem.
It runs a **SOM ring heuristic** (the visual baseline / reference, with 2-opt and
an exact optimum for small `n`) **side by side** with your **custom regions
algorithm**, on the same set of points — so you can watch, compare and export.

Built to be **extended**: algorithms, data distributions and export formats are
plugins. The GUI discovers them automatically. For the internals see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

This is the initial Git baseline, migrated from the legacy v4.1 workbench. See [CHANGELOG.md](CHANGELOG.md) and [ROADMAP.md](ROADMAP.md).

## Install & run

```bash
pip install -r requirements.txt    # numpy + matplotlib
python main.py                     # launch the GUI
```

Python 3.10+ with Tk (ships with the standard python.org installer on
Windows/macOS; on Linux: `sudo apt install python3-tk`).

## The window at a glance

* **Instance bar** — number of points `n`, `seed`, point distribution
  (uniform / clustered / circle), `Reset random`, `Clear points`, `Settings`,
  and a `Mouse` mode to edit points directly on the canvas
  (select / add / move / delete).
* **Algorithm control** — pick the active algorithm, then drive its steps:
  * a **step dropdown** that spells out each step's dependency and status, e.g.
    `Stage 2 · Filter   ·  needs: Stage 1   ·  [LOCKED]`;
  * a **colour-coded dependency strip** showing the whole chain at a glance —
    grey = locked, blue = ready, yellow = running, green = done, with `→`
    arrows and `▸` on the active step;
  * buttons `▶ Step` (one micro-step), `Run step`, `Run all`, `Reset this`, and
    `Play/Pause` with a speed slider for animation.
* **Display panel** (right) — choose which algorithm to *emphasise*, tick which
  **info fields** appear in the State panel, and toggle each **visual layer**
  (SOM ring / tours, custom regions / edges / route…).
* **State panel** — shows an **honest cost breakdown** (logical steps vs.
  distance evaluations vs. unit updates vs. wall-clock ms) instead of one
  misleading "steps" number.
* **Comparison table** — every produced tour with `length`, `% gap` vs the exact
  optimum, and cost columns.
* **Export / Save / Load** — export Nodes/Results/Tours to CSV, the map to PNG,
  or the whole session to JSON; save a full run (all algorithms + results) and
  load it back later.

## Why "steps" was misleading (and what you see now)

The old counter (`3169/5000`) only counted how many times a city was presented
to the SOM. Each such tick actually searches and updates *every* neuron, so the
real work is roughly `ticks × neurons`. The cost panel now reports the real
distance-evaluation and unit-update counts plus wall-clock time, per algorithm.

## Headless / batch use (no GUI)

```bash
# run on one instance and print the comparison table (+ optional outputs)
python cli.py run --n 12 --seed 7 --algos som custom --export results.csv --png map.png --save run.json

# average an algorithm over many seeds
python cli.py benchmark --algo custom --n 30 --seeds 0-20
```

## The custom algorithm

Stage 1 (mutual-nearest pairing) → Stage 2 (strict filter) → Stage 3
(translucent regions) → **Stage 4-7 (outline-first route construction)**:
build the perimeter (convex hull of region centres) and flag suspect
large-radius regions, insert the interior regions by cheapest insertion (order
does not matter), expand the region cycle into one Hamiltonian tour, then do a
bounded long-edge 2-opt repair. The result is a single closed loop over **all**
nodes. To change it, edit `_stage4_perimeter` … `_stage7_repair` in
`algorithms/custom_regions.py` — drawing, comparison, export and save/load
already support it. See `ARCHITECTURE.md` §4.

## Extending in one minute

* **New algorithm** → subclass `Algorithm`, declare its `STEPS`, `@register_algorithm`,
  add one import to `bootstrap.py`.
* **New data distribution** → one `@register_generator("name")` function.
* **New export format** → one `@register_exporter("Label", "ext")` function.

Nothing in the GUI needs editing. Details and a full module map are in
[`ARCHITECTURE.md`](ARCHITECTURE.md).
