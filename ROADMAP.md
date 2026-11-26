# Development roadmap

Use one release branch per version (e.g. `release/0.1.0`). Main is reserved for verified stable snapshots. Do not claim a release or merge until tests and CI pass.

- 0.1.0: recover and verify the existing workbench; retain all functional source modules.
- 0.2.0: review session compatibility, edge cases, algorithm instrumentation, benchmark selection and dependency reproducibility; expand tests.
- 0.3.0: benchmark custom regions, SOM, 2-opt and exact solutions against deterministic instances; measure accuracy, runtime, operation counts and scalability.
- 0.4.0: improve responsiveness and rendering after profiling; separate worker execution from the GUI if justified.
- 1.0.0: documented and tested stable release. Version targets may be revised after inspection and measurement.

## Baseline limitations

Headless tests do not certify interactive GUI behavior. The legacy source has no explicit license; do not assign one without the owner's decision. Check the legacy cost accounting, benchmark variant selection, zero-node behavior and reproducibility before interpreting performance claims. The on-disk session schema version remains 4 for compatibility.
