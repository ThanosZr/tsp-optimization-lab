"""The Tk workbench window.

It is generic: it discovers algorithms, their steps, their layers and their
draw code through the Algorithm interface and the registries. Adding an
algorithm / data generator / exporter requires **no change here**.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.algorithm import DONE, LOCKED, READY, RUNNING
from core.data import make_nodes
from core.models import Node
from core.registry import exporters, generator_names
from core.render import render
from core.session import Session

STATUS_COLOR = {
    LOCKED: "#d9d9d9",     # grey   – prerequisite missing
    READY: "#cfe3ff",      # blue   – ready to run
    RUNNING: "#fff1b8",    # yellow – currently animating
    DONE: "#c8efc8",       # green  – completed
}
STATUS_LABEL = {LOCKED: "LOCKED", READY: "READY", RUNNING: "RUNNING", DONE: "DONE"}


class TSPWorkbenchGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("TSP Workbench v4")
        self.geometry("1600x980")

        # ---- instance controls
        self.n_var = tk.IntVar(value=40)
        self.seed_var = tk.IntVar(value=7)
        self.data_mode_var = tk.StringVar(value="uniform")
        self.mouse_mode_var = tk.StringVar(value="select")
        self.snap_var = tk.DoubleVar(value=0.035)

        # ---- which algorithm the step-panel / Play controls drive
        self.active_algo_var = tk.StringVar(value="som")
        self.speed_var = tk.IntVar(value=30)
        self.emphasis_var = tk.StringVar(value="(none)")

        # ---- SOM params
        self.p_neuron_factor = tk.DoubleVar(value=8.0)
        self.p_iterations = tk.IntVar(value=5000)
        self.p_lr = tk.DoubleVar(value=0.8)
        self.p_radius = tk.DoubleVar(value=0.0)
        self.p_two_opt = tk.IntVar(value=50)
        self.p_exact_n = tk.IntVar(value=12)
        # ---- custom params
        self.p_min_cand = tk.IntVar(value=8)
        self.p_exact_search = tk.BooleanVar(value=False)
        self.p_filter_k = tk.IntVar(value=8)
        self.p_max_rank = tk.IntVar(value=3)
        self.p_best_ratio = tk.DoubleVar(value=2.0)
        self.p_median_ratio = tk.DoubleVar(value=1.7)
        self.p_one_side = tk.BooleanVar(value=True)
        self.p_padding = tk.DoubleVar(value=0.035)
        self.p_singleton = tk.DoubleVar(value=0.045)
        self.p_region_alpha = tk.DoubleVar(value=0.22)

        # ---- info-field toggles (which info shows in the State panel)
        self.info_vars = {
            "frame": tk.BooleanVar(value=True),       # step/phase/message
            "current": tk.BooleanVar(value=True),     # current node / highlights
            "cost": tk.BooleanVar(value=True),        # honest cost metrics
            "notes": tk.BooleanVar(value=True),       # per-frame notes
            "summary": tk.BooleanVar(value=True),     # per-algorithm summary
            "selection": tk.BooleanVar(value=True),   # selected points
            "history": tk.BooleanVar(value=True),     # recent history
        }
        self.emph_cost_var = tk.BooleanVar(value=True)

        # ---- runtime
        self.session: Optional[Session] = None
        self.layer_vars: Dict[str, tk.BooleanVar] = {}
        self.last_frame = None
        self.selected_nodes: List[int] = []
        self.drag_node: Optional[int] = None
        self.dragging = False
        self.playing = False
        self.after_id: Optional[str] = None
        self._step_index_by_id: Dict[str, int] = {}

        self.step_combo_var = tk.StringVar()

        self._build_layout()
        self.reset_random()

    # ================================================================== #
    # layout
    # ================================================================== #
    def _build_layout(self) -> None:
        left = ttk.Frame(self); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(self, width=560); right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # --- instance bar
        bar = ttk.Frame(left); bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(bar, text="n").pack(side=tk.LEFT)
        ttk.Spinbox(bar, from_=0, to=100000, textvariable=self.n_var, width=7).pack(side=tk.LEFT, padx=3)
        ttk.Label(bar, text="seed").pack(side=tk.LEFT)
        ttk.Spinbox(bar, from_=0, to=999999, textvariable=self.seed_var, width=8).pack(side=tk.LEFT, padx=3)
        ttk.Label(bar, text="data").pack(side=tk.LEFT)
        ttk.Combobox(bar, textvariable=self.data_mode_var, values=generator_names(),
                     width=10, state="readonly").pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Reset random", command=self.reset_random).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Clear points", command=self.clear_points).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Settings", command=self.open_settings).pack(side=tk.LEFT, padx=4)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(bar, text="Mouse").pack(side=tk.LEFT)
        ttk.Combobox(bar, textvariable=self.mouse_mode_var, values=["select", "add", "move", "delete"],
                     width=8, state="readonly").pack(side=tk.LEFT, padx=4)

        # --- algorithm + step control panel
        ctrl = ttk.LabelFrame(left, text="Algorithm control")
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        row1 = ttk.Frame(ctrl); row1.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Label(row1, text="Active algorithm").pack(side=tk.LEFT)
        self.algo_combo = ttk.Combobox(row1, textvariable=self.active_algo_var, width=20, state="readonly")
        self.algo_combo.pack(side=tk.LEFT, padx=5)
        self.algo_combo.bind("<<ComboboxSelected>>", lambda e: self._on_algo_change())
        ttk.Label(row1, text="Step").pack(side=tk.LEFT, padx=(12, 2))
        self.step_combo = ttk.Combobox(row1, textvariable=self.step_combo_var, width=46, state="readonly")
        self.step_combo.pack(side=tk.LEFT, padx=4)
        self.step_combo.bind("<<ComboboxSelected>>", lambda e: self._on_step_select())

        # color-coded dependency strip
        self.step_strip = ttk.Frame(ctrl); self.step_strip.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)

        row2 = ttk.Frame(ctrl); row2.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        for txt, cmd in [("▶ Step", self.do_step), ("Run step", self.do_run_step),
                         ("Run all", self.do_run_all), ("Reset this", self.do_reset_algo)]:
            ttk.Button(row2, text=txt, command=cmd).pack(side=tk.LEFT, padx=3)
        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.play_btn = ttk.Button(row2, text="Play", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=3)
        ttk.Label(row2, text="speed ms").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Scale(row2, from_=1, to=400, variable=self.speed_var, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT)

        # --- canvas
        self.fig = Figure(figsize=(8.6, 7.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        # ============ right column ============
        disp = ttk.LabelFrame(right, text="Display")
        disp.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))
        em = ttk.Frame(disp); em.pack(side=tk.TOP, fill=tk.X, padx=6, pady=3)
        ttk.Label(em, text="Emphasise").pack(side=tk.LEFT)
        self.emph_combo = ttk.Combobox(em, textvariable=self.emphasis_var, width=18, state="readonly")
        self.emph_combo.pack(side=tk.LEFT, padx=5)
        self.emph_combo.bind("<<ComboboxSelected>>", lambda e: self.draw())
        ttk.Checkbutton(em, text="emphasise cost", variable=self.emph_cost_var,
                        command=self.update_state).pack(side=tk.LEFT, padx=8)

        infofrm = ttk.Frame(disp); infofrm.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)
        ttk.Label(infofrm, text="Show info:").pack(side=tk.LEFT)
        for key, lbl in [("frame", "frame"), ("current", "current"), ("cost", "cost"),
                         ("notes", "notes"), ("summary", "summary"),
                         ("selection", "selection"), ("history", "history")]:
            ttk.Checkbutton(infofrm, text=lbl, variable=self.info_vars[key],
                            command=self.update_state).pack(side=tk.LEFT)

        self.layers_frame = ttk.Frame(disp); self.layers_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=3)

        ttk.Label(right, text="State", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        self.state_text = tk.Text(right, height=18, wrap=tk.WORD)
        self.state_text.pack(fill=tk.X, padx=8, pady=2)
        self.state_text.tag_configure("emph", font=("TkDefaultFont", 10, "bold"), foreground="#7b241c")

        cmp = ttk.LabelFrame(right, text="Comparison"); cmp.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)
        cols = ("algorithm", "variant", "length", "gap%", "steps", "dist_evals", "ms")
        self.tree = ttk.Treeview(cmp, columns=cols, show="headings", height=7)
        widths = {"algorithm": 100, "variant": 130, "length": 70, "gap%": 60, "steps": 60, "dist_evals": 80, "ms": 60}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths[c], anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        io = ttk.Frame(right); io.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)
        ttk.Label(io, text="Export").pack(side=tk.LEFT)
        self.export_var = tk.StringVar(value=list(exporters().keys())[0])
        ttk.Combobox(io, textvariable=self.export_var, values=list(exporters().keys()),
                     width=20, state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Button(io, text="Export", command=self.do_export).pack(side=tk.LEFT, padx=4)
        ttk.Separator(io, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(io, text="Save run", command=self.do_save).pack(side=tk.LEFT, padx=4)
        ttk.Button(io, text="Load run", command=self.do_load).pack(side=tk.LEFT, padx=4)

    # ================================================================== #
    # params / session lifecycle
    # ================================================================== #
    def _collect_params(self) -> Dict[str, dict]:
        seed = int(self.seed_var.get())
        return {
            "som": {"seed": seed, "neuron_factor": float(self.p_neuron_factor.get()),
                    "som_iterations": int(self.p_iterations.get()),
                    "learning_rate": float(self.p_lr.get()),
                    "radius": float(self.p_radius.get()),
                    "two_opt_passes": int(self.p_two_opt.get()),
                    "exact_max_n": int(self.p_exact_n.get())},
            "custom": {"seed": seed, "custom_min_candidates": int(self.p_min_cand.get()),
                       "custom_exact_search": bool(self.p_exact_search.get()),
                       "custom_filter_k": int(self.p_filter_k.get()),
                       "custom_max_rank": int(self.p_max_rank.get()),
                       "custom_max_best_ratio": float(self.p_best_ratio.get()),
                       "custom_max_median_ratio": float(self.p_median_ratio.get()),
                       "custom_reject_one_side": bool(self.p_one_side.get()),
                       "custom_region_padding": float(self.p_padding.get()),
                       "custom_singleton_radius": float(self.p_singleton.get()),
                       "custom_region_alpha": float(self.p_region_alpha.get())},
        }

    def reset_random(self) -> None:
        self.stop_play()
        nodes = make_nodes(int(self.n_var.get()), seed=int(self.seed_var.get()), mode=self.data_mode_var.get())
        self.session = Session(nodes, seed=int(self.seed_var.get()),
                               data_mode=self.data_mode_var.get(), params=self._collect_params())
        self.selected_nodes = []
        self.last_frame = None
        self.session.log("Reset random points.")
        self._refresh_algo_lists()
        self._rebuild_layer_toggles()
        self._on_algo_change()
        self.update_state(); self.draw()

    def clear_points(self) -> None:
        self.stop_play()
        self.session = Session([], seed=int(self.seed_var.get()),
                               data_mode=self.data_mode_var.get(), params=self._collect_params())
        self.selected_nodes = []; self.last_frame = None; self.n_var.set(0)
        self.session.log("Cleared all points.")
        self._rebuild_layer_toggles(); self.update_state(); self.draw()

    def apply_settings(self) -> None:
        """Rebuild algorithms with current params (loses computed results)."""
        self.stop_play()
        self.session.rebuild_algorithms(self._collect_params())
        self.last_frame = None
        self.session.log("Settings applied; algorithms rebuilt.")
        self._on_algo_change(); self.update_state(); self.draw()

    def _structural_change(self, reason: str) -> None:
        """Nodes changed -> algorithm results no longer valid; rebuild fresh."""
        self.stop_play()
        self.session.set_nodes(self.session.nodes)  # rebuild against current nodes
        self.session.rebuild_algorithms(self._collect_params())
        self.last_frame = None
        self.session.log(reason)
        self._on_algo_change()

    # ================================================================== #
    # algorithm / step UI
    # ================================================================== #
    def _refresh_algo_lists(self) -> None:
        algos = self.session.algos()
        ids = [a.id for a in algos]
        names = [f"{a.name}  [{a.id}]" for a in algos]
        self._algo_id_by_label = dict(zip(names, ids))
        self.algo_combo["values"] = names
        if self.active_algo_var.get() not in ids:
            self.active_algo_var.set(ids[0] if ids else "")
        # set combo display to the label matching the active id
        for lbl, aid in self._algo_id_by_label.items():
            if aid == self.active_algo_var.get():
                self.algo_combo.set(lbl)
        self.emph_combo["values"] = ["(none)"] + names

    def _active_algo(self):
        return self.session.algorithms.get(self.active_algo_var.get())

    def _on_algo_change(self) -> None:
        # resolve label -> id (combo shows labels)
        lbl = self.algo_combo.get()
        if hasattr(self, "_algo_id_by_label") and lbl in self._algo_id_by_label:
            self.active_algo_var.set(self._algo_id_by_label[lbl])
        self._refresh_step_combo()
        self._refresh_step_strip()
        self.update_state(); self.draw()

    def _step_label(self, algo, step) -> str:
        st = algo.status(step.id)
        reqs = ", ".join(self._step_disp_name(algo, r) for r in step.requires) or "—"
        return f"{step.label}   ·  needs: {reqs}   ·  [{STATUS_LABEL[st]}]"

    def _step_disp_name(self, algo, step_id) -> str:
        try:
            return algo._step(step_id).label
        except Exception:
            return step_id

    def _refresh_step_combo(self) -> None:
        algo = self._active_algo()
        if not algo:
            self.step_combo["values"] = []; return
        labels = [self._step_label(algo, s) for s in algo.steps()]
        self._step_index_by_id = {s.id: i for i, s in enumerate(algo.steps())}
        self.step_combo["values"] = labels
        active = algo.active_step_id
        if active is not None and active in self._step_index_by_id:
            self.step_combo.current(self._step_index_by_id[active])

    def _refresh_step_strip(self) -> None:
        for w in self.step_strip.winfo_children():
            w.destroy()
        algo = self._active_algo()
        if not algo:
            return
        for i, s in enumerate(algo.steps()):
            st = algo.status(s.id)
            if i > 0:
                tk.Label(self.step_strip, text="→").pack(side=tk.LEFT, padx=1)
            txt = f" {s.label} "
            if s.id == algo.active_step_id:
                txt = f"▸{s.label} "
            lbl = tk.Label(self.step_strip, text=txt, bg=STATUS_COLOR[st],
                           relief="ridge", bd=1, padx=4, pady=1, cursor="hand2")
            lbl.pack(side=tk.LEFT, padx=1)
            lbl.bind("<Button-1>", lambda e, sid=s.id: self._select_step(sid))

    def _select_step(self, step_id: str) -> None:
        algo = self._active_algo()
        if not algo:
            return
        if algo.status(step_id) == LOCKED:
            self.session.log(f"Step '{step_id}' is locked.")
        else:
            algo.active_step_id = step_id
        self._refresh_step_combo(); self._refresh_step_strip(); self.update_state()

    def _on_step_select(self) -> None:
        algo = self._active_algo()
        if not algo:
            return
        idx = self.step_combo.current()
        steps = algo.steps()
        if 0 <= idx < len(steps):
            self._select_step(steps[idx].id)

    # ================================================================== #
    # run actions
    # ================================================================== #
    def _post_run(self, frames) -> None:
        if frames:
            self.last_frame = frames[-1]
        self._refresh_step_combo(); self._refresh_step_strip()
        self.update_state(); self.draw()

    def do_step(self) -> None:
        algo = self._active_algo()
        if not algo:
            return
        f = algo.micro_step()
        self._post_run([f] if f else [])

    def do_run_step(self) -> None:
        algo = self._active_algo()
        if not algo:
            return
        sid = algo.active_step_id
        frames = algo.run_step(sid)
        self.session.log(f"{algo.name}: ran step '{sid}' ({len(frames)} frames).")
        self._post_run(frames)

    def do_run_all(self) -> None:
        algo = self._active_algo()
        if not algo:
            return
        frames = algo.run_all()
        self.session.log(f"{algo.name}: ran all ({len(frames)} frames).")
        self._post_run(frames)

    def do_reset_algo(self) -> None:
        algo = self._active_algo()
        if not algo:
            return
        self.stop_play()
        f = algo.reset()
        self.last_frame = f
        self.session.log(f"{algo.name} reset.")
        self._post_run([f])

    def toggle_play(self) -> None:
        if self.playing:
            self.stop_play()
        else:
            self.playing = True
            self.play_btn.config(text="Pause")
            self._tick()

    def stop_play(self) -> None:
        self.playing = False
        if hasattr(self, "play_btn"):
            self.play_btn.config(text="Play")
        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def _tick(self) -> None:
        if not self.playing:
            return
        algo = self._active_algo()
        if algo:
            f = algo.micro_step()
            if f is not None:
                self.last_frame = f
            if algo.is_done():
                self.stop_play()
            self._refresh_step_combo(); self._refresh_step_strip()
            self.update_state(); self.draw()
        self.after_id = self.after(int(float(self.speed_var.get())), self._tick)

    # ================================================================== #
    # layers
    # ================================================================== #
    def _rebuild_layer_toggles(self) -> None:
        for w in self.layers_frame.winfo_children():
            w.destroy()
        self.layer_vars = {}
        if not self.session:
            return
        ttk.Label(self.layers_frame, text="Layers:").grid(row=0, column=0, sticky="w")
        col = 1
        row = 0
        for algo in self.session.algos():
            ttk.Label(self.layers_frame, text=algo.name, font=("TkDefaultFont", 9, "bold")
                      ).grid(row=row, column=col, sticky="w", padx=(8, 2))
            col += 1
            for ls in algo.layers():
                key = f"{algo.id}:{ls.id}"
                var = tk.BooleanVar(value=ls.default_on)
                self.layer_vars[key] = var
                ttk.Checkbutton(self.layers_frame, text=ls.label, variable=var,
                                command=self.draw).grid(row=row, column=col, sticky="w")
                col += 1
                if col > 7:
                    col = 1; row += 1

    def _layer_on(self) -> Dict[str, bool]:
        return {k: v.get() for k, v in self.layer_vars.items()}

    # ================================================================== #
    # state panel
    # ================================================================== #
    def update_state(self, prefix: str = "") -> None:
        if not self.session:
            return
        t = self.state_text
        t.delete("1.0", tk.END)
        iv = self.info_vars
        algo = self._active_algo()

        if prefix:
            t.insert(tk.END, prefix + "\n\n")

        # cost up top if emphasised
        if iv["cost"].get() and self.emph_cost_var.get() and algo:
            self._insert_cost(algo, emph=True)

        if iv["frame"].get() and self.last_frame:
            f = self.last_frame
            t.insert(tk.END, f"[{f.algo_id}] step={f.step_id} phase={f.phase}\n{f.message}\n")
        if iv["current"].get() and self.last_frame and self.last_frame.highlight_nodes:
            t.insert(tk.END, f"highlight nodes: {self.last_frame.highlight_nodes}\n")
        if iv["notes"].get() and self.last_frame and self.last_frame.notes:
            for k, v in self.last_frame.notes.items():
                t.insert(tk.END, f"  {k}: {v}\n")
        t.insert(tk.END, "\n")

        if iv["cost"].get() and not self.emph_cost_var.get() and algo:
            self._insert_cost(algo, emph=False)

        t.insert(tk.END, f"Active algorithm: {algo.name if algo else '—'}\n")
        t.insert(tk.END, f"Cities: {len(self.session.nodes)}   Mouse: {self.mouse_mode_var.get()}\n")

        if iv["selection"].get():
            t.insert(tk.END, self._selection_text())

        if iv["summary"].get():
            t.insert(tk.END, "\nPer-algorithm:\n")
            for a in self.session.algos():
                done = sum(1 for s in a.steps() if a.is_step_done(s.id))
                t.insert(tk.END, f"  {a.name}: {done}/{len(a.steps())} steps · "
                                 f"{a.cost.logical_steps} ticks · "
                                 f"{a.cost.distance_evals} dist-evals · {a.cost.elapsed_ms:.1f} ms\n")

        if iv["history"].get():
            t.insert(tk.END, "\nRecent history:\n")
            for h in self.session.history[-8:]:
                t.insert(tk.END, f"- {h}\n")

        self._refresh_tree()

    def _insert_cost(self, algo, emph: bool) -> None:
        t = self.state_text
        tag = ("emph",) if emph else ()
        c = algo.cost
        t.insert(tk.END, "Cost (honest):\n", tag)
        t.insert(tk.END, f"  logical steps : {c.logical_steps}\n", tag)
        t.insert(tk.END, f"  distance evals: {c.distance_evals}\n", tag)
        t.insert(tk.END, f"  unit updates  : {c.unit_updates}\n", tag)
        t.insert(tk.END, f"  wall time     : {c.elapsed_ms:.1f} ms\n", tag)
        t.insert(tk.END, "\n")

    def _selection_text(self) -> str:
        if not self.selected_nodes:
            return "Selected: none\n"
        by_id = self.session.algorithms[next(iter(self.session.algorithms))].node_by_id() \
            if self.session.algorithms else {n.idx: n for n in self.session.nodes}
        by_id = {n.idx: n for n in self.session.nodes}
        lines = ["Selected:"]
        for idx in self.selected_nodes:
            if idx in by_id:
                p = by_id[idx]; lines.append(f"  {idx}: ({p.x:.4f}, {p.y:.4f})")
        if len(self.selected_nodes) == 2:
            a, b = self.selected_nodes
            if a in by_id and b in by_id:
                lines.append(f"Distance {a}-{b}: {math.hypot(by_id[a].x-by_id[b].x, by_id[a].y-by_id[b].y):.5f}")
        return "\n".join(lines) + "\n"

    def _refresh_tree(self) -> None:
        for r in self.tree.get_children():
            self.tree.delete(r)
        for row in self.session.comparison_rows():
            L = row.get("length")
            gap = row.get("gap_pct")
            self.tree.insert("", tk.END, values=(
                row.get("algorithm", ""), row.get("variant", ""),
                f"{L:.4f}" if isinstance(L, (int, float)) else "—",
                f"{gap:.2f}" if isinstance(gap, (int, float)) else "—",
                row.get("steps", ""), row.get("dist_evals", ""), row.get("ms", ""),
            ))

    # ================================================================== #
    # drawing
    # ================================================================== #
    def draw(self) -> None:
        if not self.session:
            return
        emph_lbl = self.emphasis_var.get()
        emph_id = None
        if hasattr(self, "_algo_id_by_label") and emph_lbl in self._algo_id_by_label:
            emph_id = self._algo_id_by_label[emph_lbl]
        cur = self.last_frame.highlight_nodes if self.last_frame else []
        render(self.ax, self.session.nodes, self.session.algos(),
               layer_on=self._layer_on(), emphasis_algo=emph_id,
               selected=self.selected_nodes, current_highlight=cur,
               title="TSP Workbench v4")
        self.canvas.draw_idle()

    # ================================================================== #
    # mouse editing
    # ================================================================== #
    def _nearest(self, x, y) -> Optional[int]:
        best = None
        for p in self.session.nodes:
            d = math.hypot(p.x - x, p.y - y)
            if best is None or d < best[0]:
                best = (d, p.idx)
        return best[1] if best and best[0] <= float(self.snap_var.get()) else None

    def on_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        mode = self.mouse_mode_var.get()
        idx = self._nearest(x, y)
        if mode == "add" and idx is None:
            self._add_point(x, y)
        elif mode == "delete" and idx is not None:
            self._delete_point(idx)
        elif mode == "move" and idx is not None:
            self.drag_node = idx; self.dragging = True; self.selected_nodes = [idx]
            self._move_point(idx, x, y, final=False)
        else:  # select
            if idx is not None:
                if idx in self.selected_nodes:
                    self.selected_nodes.remove(idx)
                else:
                    self.selected_nodes.append(idx)
                    self.selected_nodes = self.selected_nodes[-8:]
                self.update_state(); self.draw()

    def on_motion(self, event) -> None:
        if self.dragging and self.drag_node is not None and event.inaxes == self.ax and event.xdata is not None:
            self._move_point(self.drag_node, float(event.xdata), float(event.ydata), final=False)

    def on_release(self, event) -> None:
        if self.dragging and self.drag_node is not None:
            if event.inaxes == self.ax and event.xdata is not None:
                self._move_point(self.drag_node, float(event.xdata), float(event.ydata), final=True)
            else:
                self._structural_change(f"Moved point {self.drag_node}.")
                self.update_state(); self.draw()
        self.dragging = False; self.drag_node = None

    def _add_point(self, x, y) -> None:
        x = min(1.0, max(0.0, x)); y = min(1.0, max(0.0, y))
        nodes = list(self.session.nodes)
        new_id = len(nodes)
        nodes.append(Node(new_id, x, y))
        self.session.nodes = nodes
        self.selected_nodes = [new_id]; self.n_var.set(len(nodes))
        self._structural_change(f"Added point {new_id}.")
        self.update_state(); self.draw()

    def _delete_point(self, idx) -> None:
        nodes = [p for p in self.session.nodes if p.idx != idx]
        nodes = [Node(i, p.x, p.y) for i, p in enumerate(nodes)]  # reindex
        self.session.nodes = nodes
        self.selected_nodes = []
        self.n_var.set(len(nodes))
        self._structural_change(f"Deleted point {idx}; reindexed.")
        self.update_state(); self.draw()

    def _move_point(self, idx, x, y, final=False) -> None:
        x = min(1.0, max(0.0, x)); y = min(1.0, max(0.0, y))
        self.session.nodes = [Node(p.idx, x, y) if p.idx == idx else p for p in self.session.nodes]
        if final:
            self._structural_change(f"Moved point {idx}.")
        self.update_state(); self.draw()

    # ================================================================== #
    # export / save / load
    # ================================================================== #
    def do_export(self) -> None:
        label = self.export_var.get()
        spec = exporters()[label]
        path = filedialog.asksaveasfilename(defaultextension="." + spec["ext"],
                                            filetypes=[(label, "*." + spec["ext"]), ("All", "*.*")])
        if not path:
            return
        try:
            if label == "Map (PNG)":
                spec["fn"](self.session, path, layer_on=self._layer_on())
            else:
                spec["fn"](self.session, path)
            self.session.log(f"Exported '{label}' -> {path}")
            self.update_state()
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def do_save(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("Session", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            self.session.params = self._collect_params()
            self.session.save(path)
            self.session.log(f"Saved run -> {path}")
            self.update_state()
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def do_load(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Session", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            self.stop_play()
            self.session = Session.load(path)
            self.seed_var.set(self.session.seed)
            self.data_mode_var.set(self.session.data_mode)
            self.n_var.set(len(self.session.nodes))
            self.selected_nodes = []; self.last_frame = None
            self._apply_params_to_vars(self.session.params)
            self._refresh_algo_lists(); self._rebuild_layer_toggles(); self._on_algo_change()
            self.session.log(f"Loaded run <- {path}")
            self.update_state(); self.draw()
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _apply_params_to_vars(self, params: Dict[str, dict]) -> None:
        som = params.get("som", {}); cus = params.get("custom", {})
        self.p_neuron_factor.set(som.get("neuron_factor", self.p_neuron_factor.get()))
        self.p_iterations.set(som.get("som_iterations", self.p_iterations.get()))
        self.p_lr.set(som.get("learning_rate", self.p_lr.get()))
        self.p_radius.set(som.get("radius", self.p_radius.get()))
        self.p_two_opt.set(som.get("two_opt_passes", self.p_two_opt.get()))
        self.p_exact_n.set(som.get("exact_max_n", self.p_exact_n.get()))
        self.p_min_cand.set(cus.get("custom_min_candidates", self.p_min_cand.get()))
        self.p_exact_search.set(cus.get("custom_exact_search", self.p_exact_search.get()))
        self.p_filter_k.set(cus.get("custom_filter_k", self.p_filter_k.get()))
        self.p_max_rank.set(cus.get("custom_max_rank", self.p_max_rank.get()))
        self.p_best_ratio.set(cus.get("custom_max_best_ratio", self.p_best_ratio.get()))
        self.p_median_ratio.set(cus.get("custom_max_median_ratio", self.p_median_ratio.get()))
        self.p_one_side.set(cus.get("custom_reject_one_side", self.p_one_side.get()))
        self.p_padding.set(cus.get("custom_region_padding", self.p_padding.get()))
        self.p_singleton.set(cus.get("custom_singleton_radius", self.p_singleton.get()))
        self.p_region_alpha.set(cus.get("custom_region_alpha", self.p_region_alpha.get()))

    # ================================================================== #
    # settings
    # ================================================================== #
    def open_settings(self) -> None:
        win = tk.Toplevel(self); win.title("Settings"); win.geometry("640x640")
        frm = ttk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        r = 0
        ttk.Label(frm, text="SOM", font=("TkDefaultFont", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 8)); r += 1
        for lbl, var in [("Neuron factor", self.p_neuron_factor), ("SOM iterations", self.p_iterations),
                         ("Learning rate", self.p_lr), ("Radius (0=auto)", self.p_radius),
                         ("2-opt passes", self.p_two_opt), ("Exact max n", self.p_exact_n)]:
            ttk.Label(frm, text=lbl).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(frm, textvariable=var, width=12).grid(row=r, column=1, sticky="w", pady=3); r += 1
        ttk.Label(frm, text="Custom regions", font=("TkDefaultFont", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", pady=(14, 8)); r += 1
        for lbl, var in [("Stage1 min candidates", self.p_min_cand), ("Stage2 filter k", self.p_filter_k),
                         ("Stage2 max rank", self.p_max_rank), ("Stage2 max edge/best", self.p_best_ratio),
                         ("Stage2 max edge/median", self.p_median_ratio), ("Region padding", self.p_padding),
                         ("Singleton radius", self.p_singleton), ("Region alpha", self.p_region_alpha),
                         ("Click snap threshold", self.snap_var)]:
            ttk.Label(frm, text=lbl).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(frm, textvariable=var, width=12).grid(row=r, column=1, sticky="w", pady=3); r += 1
        ttk.Checkbutton(frm, text="Custom exact local search", variable=self.p_exact_search).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Checkbutton(frm, text="Stage2 reject if bad from one side", variable=self.p_one_side).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Apply (rebuild)", command=lambda: (self.apply_settings(), win.destroy())).grid(row=r, column=0, sticky="w", pady=12)
        ttk.Button(frm, text="Cancel", command=win.destroy).grid(row=r, column=1, sticky="w", pady=12)

    def on_close(self) -> None:
        self.stop_play(); self.destroy()
