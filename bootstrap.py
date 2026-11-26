"""Import this once to populate every registry (data generators, algorithms,
exporters). Both main.py and cli.py do `import bootstrap` before building a
Session, so registration always happens first."""
from __future__ import annotations

# Data generators self-register on import.
import core.data            # noqa: F401
# Exporters self-register on import.
import core.export          # noqa: F401
# Algorithms self-register via @register_algorithm on import.
import algorithms.som       # noqa: F401
import algorithms.custom_regions  # noqa: F401
