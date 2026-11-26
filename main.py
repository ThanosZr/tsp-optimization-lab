"""Launch the TSP Workbench GUI:  python main.py"""
from __future__ import annotations

import bootstrap  # noqa: F401  (registers generators, algorithms, exporters)
from gui.app import TSPWorkbenchGUI


def main() -> None:
    app = TSPWorkbenchGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
