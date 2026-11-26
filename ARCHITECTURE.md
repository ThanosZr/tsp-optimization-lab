# TSP Workbench v4 — ARCHITECTURE

> Single source of truth. Read this first before changing code. It is meant to
> stay current so future edits do **not** require re-reading every file.
> When you change behaviour, update the matching section here and bump the
> Changelog at the bottom.

## 1. What this program is

A desktop workbench (tkinter + matplotlib) that runs, animates and compares TSP
algorithms on the same point cloud:

* **SOM baseline** (`som`) — a Self-Organising-Map ring heuristic used as the
  *visual reference*, plus its 2-opt refinement and an exact Held-Karp optimum
  for small `n`.
* **Custom regions** (`custom`) — your own algorithm: Stage 1 mutual-nearest
  pairing → Stage 2 strict filter → Stage 3 translucent regions → Stage 4-7
  outline-first route construction (perimeter → insert interior → expand →
  repair) producing a single Hamiltonian cycle.

The whole thing is a **plugin system**: the GUI never names an algorithm. It
discovers algorithms, their steps, their drawable layers and their results
through the interfaces below.

## 2. Module map

```
main.py                 GUI entry point (import bootstrap; launch window)
cli.py                  headless runner: `run`, `benchmark`, export, save
bootstrap.py            imports the side-effect modules so registries fill up

core/
  models.py             Node, geometry (distance/tour_edges/tour_length), TourResult
  algorithm.py          THE CONTRACT: Step, CostMeter, VisualFrame, LayerSpec,
                        Algorithm base class, DrawContext, status constants
  registry.py           register/lookup for algorithms, generators, exporters
  data.py               point generators (uniform / clustered / circle)
  render.py             draw a node cloud + every algorithm onto an Axes
  session.py            Session: one instance + all algorithms; save/load JSON
  export.py             exporters (Nodes CSV, Results CSV, Tours CSV, Map PNG, Session JSON)

algorithms/
  tsp_utils.py          two_opt + held_karp_exact (both take an optional CostMeter)
  som.py                SOMAlgorithm  (id="som")
  custom_regions.py     CustomRegionsAlgorithm (id="custom") + RegionComponent

gui/
  app.py                TSPWorkbenchGUI — generic over registered algorithms
```

Dependency direction: `gui` and `cli` depend on `core` + `algorithms`; `core`
never imports `gui`. `render.py` only needs matplotlib + the Algorithm contract,
so both the GUI canvas and the PNG exporter reuse it.

## 3. The Algorithm contract (`core/algorithm.py`)

Every algorithm subclasses `Algorithm` and is decorated with
`@register_algorithm`. Required class attributes:

| attribute | meaning |
|-----------|---------|
| `id`      | unique short string (e.g. `"som"`) used as a key everywhere |
| `name`    | human label shown in the UI |
| `color`   | base colour family for this algorithm's overlays |
| `STEPS`   | ordered list of `Step` describing the work + dependencies |

### 3.1 Steps and the dependency graph

```python
Step(id, label, requires=[...], produces=[...], description="", rerunnable=False)
```

`requires` lists the **step ids that must be DONE first**. This is exactly what
the GUI renders as the colour-coded dependency strip and the step dropdown.

`status(step_id)` returns one of:

* `LOCKED`  — a prerequisite is not done yet        (grey)
* `READY`   — prerequisites satisfied, runnable       (blue)
* `RUNNING` — currently animating                     (yellow)
* `DONE`    — completed                               (green)

`rerunnable=True` (e.g. SOM `exact`, custom `stage7`) means: after completion it
is still offered as `READY` so you can run it again; `run_step` clears its done
flag before re-running. `run_all` still runs every step exactly once, in
dependency order, including rerunnable ones (so the exact optimum is computed).

### 3.2 Execution methods

* `micro_step() -> VisualFrame | None` — advance the **finest animatable unit**
  (one SOM city presentation, one custom candidate check). Used by `Play` and
  the `▶ Step` button. Auto-advances the active step cursor when a step finishes.
* `run_step(step_id) -> [VisualFrame]` — run one step to completion.
* `run_all() -> [VisualFrame]` — run all remaining steps in dependency order.
* `reset() -> VisualFrame` — clear all state + cost.

`active_step_id` is the cursor the GUI's dropdown/strip reflects and the buttons
act on.

### 3.3 Honest cost — `CostMeter`

Logical "steps" are **not** computational cost. Each algorithm updates a
`self.cost` `CostMeter` with:

* `logical_steps`  — algorithm-defined ticks (what the old UI mislabelled "steps")
* `distance_evals` — pairwise distance computations actually performed
* `unit_updates`   — element updates (e.g. neuron moves)
* `elapsed_ms`     — wall-clock time (use `cost.start()/stop()` around real work)

Example: one SOM training tick does `m` distance evals (BMU search) **and** `m`
neuron updates, where `m` is the neuron count. So 5000 ticks with 320 neurons is
~3.2M operations — the meter shows that; the tick count alone hid it.

### 3.4 Results, serialisation, comparison

* `result() -> dict` — JSON-serialisable snapshot (tours, regions, cost…).
* `load_result(dict)` — restore from a saved session.
* `comparison_rows() -> [dict]` — one row per produced tour/structure for the
  comparison table. Keys: `algorithm, variant, kind, length(None ok), n_nodes,
  steps, dist_evals, ms, extra`. `kind="optimal"` marks the reference used to
  compute `gap_pct` for every other row (done in `Session.comparison_rows`).

### 3.5 Drawing — algorithms draw themselves

* `layers() -> [LayerSpec(id, label, default_on)]` — the toggleable visual
  layers; the GUI builds a checkbox per layer automatically.
* `draw(ax, ctx: DrawContext)` — render overlays. `ctx.layer_on[layer_id]` says
  whether each of *this algorithm's* layers is visible; `ctx.z_base` is the
  z-order floor reserved for this algorithm (10 units per algorithm).

Because algorithms draw themselves, adding one needs **zero** changes in the GUI.

## 4. How to extend

### Add a new algorithm
1. Create `algorithms/myalgo.py`.
2. Subclass `Algorithm`, set `id/name/color/STEPS`.
3. Implement `micro_step` (and optionally override `run_step`), `result`,
   `load_result`, `comparison_rows`, `layers`, `draw`.
4. Decorate the class with `@register_algorithm`.
5. Add `import algorithms.myalgo` to `bootstrap.py`.
   Done — it appears in the GUI algorithm dropdown, the layers panel, the
   comparison table, export and save/load.

### Add a step to the custom algorithm
1. Append a `Step(...)` to `CustomRegionsAlgorithm.STEPS` with the right
   `requires`.
2. Add a generator method (mirror `_stage3_regions`) and wire it into the
   `{...}[sid]()` dispatch dict in `micro_step`.
3. If it produces a tour, set `self.tour` and add a `comparison_rows` entry; add
   a `LayerSpec` + draw branch if it has new visuals.
   The dependency strip/dropdown update themselves from `STEPS`.

> Stage 4-7 implement the **outline-first** route construction:
> `stage4` builds the **perimeter** (convex hull of region centres) and flags
> suspect large-radius regions; `stage5` attaches the interior regions by
> **cheapest insertion** (order-independent); `stage6` **expands** the region
> cycle into one Hamiltonian node tour (2-node regions are oriented to chain
> smoothly, completeness enforced); `stage7` does a bounded **long-edge 2-opt
> repair**. To change the construction, edit `_stage4_perimeter` …
> `_stage7_repair`; everything downstream (the `perimeter`/`route` layers,
> comparison, export, save/load) already supports it.

### Add a data generator
Decorate a `f(n, seed) -> list[Node]` with `@register_generator("name")` in
`core/data.py`. It shows up in the `data` dropdown automatically.

### Add an exporter
Decorate `f(session, path)` with `@register_exporter("Label", "ext")` in
`core/export.py`. It appears in the export dropdown. (PNG-style exporters may
accept extra kwargs like `layer_on`; the GUI passes `layer_on` for "Map (PNG)".)

## 5. Save / load format (`SESSION_VERSION = 4`)

JSON written by `Session.save`:

```json
{
  "version": 4,
  "saved_at": "ISO-8601",
  "instance": { "n": int, "seed": int, "data_mode": str,
                "nodes": [[x, y], ...] },
  "params":  { "som": {...}, "custom": {...} },
  "results": { "som": <som.result()>, "custom": <custom.result()> },
  "history": [ "...", ... ]
}
```

`Session.from_dict` rebuilds nodes, recreates each registered algorithm with its
saved params, and calls `load_result`. Unknown algorithm ids in `results` are
ignored, so old saves still load after you add algorithms.

## 6. GUI overview (`gui/app.py`)

* **Instance bar** — n, seed, data mode, Reset random, Clear points, Settings,
  Mouse mode (select / add / move / delete on the canvas).
* **Algorithm control** — active-algorithm dropdown; a **Step dropdown** whose
  entries read `label · needs: … · [STATUS]`; a **colour-coded dependency strip**
  (`▸active`, `→` arrows, status colours); buttons `▶ Step / Run step / Run all /
  Reset this`; `Play/Pause` + speed.
* **Display panel** — *Emphasise* an algorithm; *emphasise cost*; **info-field
  checkboxes** (frame / current / cost / notes / summary / selection / history)
  that select what the State panel shows; **per-algorithm layer toggles** built
  from `layers()`.
* **State panel** — the selected info, with the honest `CostMeter` breakdown.
* **Comparison table** — `algorithm | variant | length | gap% | steps |
  dist_evals | ms` from `Session.comparison_rows`.
* **Export / Save / Load** — exporter dropdown + button; Save run / Load run.

Structural node edits (add/move/delete) rebuild fresh algorithm instances
because indices/coords changed.

## 7. CLI (`cli.py`)

* `python cli.py run --n 12 --seed 7 --algos som custom --export r.csv --png m.png --save s.json`
* `python cli.py benchmark --algo custom --n 30 --seeds 0-20`

Same core, so any registered algorithm/exporter is available headless.

## 8. Conventions

* `core` stays framework-free except `render.py`/`export.py` (matplotlib only).
* Node ids are contiguous `0..n-1`; deleting a node reindexes the rest.
* Colours: SOM = blue family, Custom = red/warm family — keep new algorithms in
  visually distinct families.

## Changelog

* **v4.1** — Custom algorithm Stage 4 replaced by an outline-first route
  builder split into four steps: `stage4` perimeter (convex hull of region
  centres + suspect-region flagging), `stage5` cheapest-insertion of interior
  regions, `stage6` region-cycle → single Hamiltonian tour, `stage7` bounded
  long-edge 2-opt repair. New `perimeter` layer; `region_cycle`, `perimeter`,
  `suspect_regions` serialised. Result: valid single loop over all nodes, mean
  gap vs exact ≈0.4% at n=11 for a few hundred distance evals (vs the old
  scaffold's ~21%).

* **v4** — Full rebuild. Plugin architecture (registries + `Algorithm`
  contract), explicit step-dependency graph with colour coding, honest
  `CostMeter`, per-algorithm self-drawing + layer toggles, selectable info
  fields, multi-algorithm `Session` save/load, exporters (CSV/PNG/JSON),
  comparison table, CLI runner + benchmark, numpy-vectorised SOM, Stage-4
  route-construction scaffold.
* **v3** — SOM baseline + custom Stage 1–3, SOM-coupled state, single monolithic
  GUI. (Superseded.)
