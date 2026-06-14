# dualbranch_baseline_50ep

## Why

Validate whether the line-two RF + baseline B-mode dual-input supervised baseline can train end to end on the local cache, with stable loss, stable memory, and required per-epoch probes.

## Single-Variable Hypothesis

Adding baseline B-mode as a second input gives the network structural guidance beyond RF-only input while keeping the rest of the route pure supervised: same cache, same loss family, no discriminator, no AMP.

## Expected

- The 3-epoch smoke run completes without NaN, OOM, or notebook errors.
- `stability.csv` is appended once per epoch.
- Train and val probe images are written every epoch.
- The 50-epoch run can proceed after smoke without changing frozen config.
- Metrics and probe images are reported as observations only; image quality is not accepted before human Slicer review.

## Final Status

TBD after run.
