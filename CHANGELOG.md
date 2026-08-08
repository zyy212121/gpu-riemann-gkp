# Changelog

All notable user-facing changes to GPU-Riemann-GKP are recorded here.

## [2.8.0] - 2026-08-08

### Added

- Explicit GPU-resident k-omega SST turbulence transport.
- `lowRe` and `wallFunction` SST wall treatments.
- OpenFOAM write/restart support for `k`, `omega`, and `nut`.
- Conservative mobile-particle packing projection with a non-negative complementarity multiplier.
- Runtime control `gpuResidentPackingProjectionIterations`, which selects both the Jacobi iteration count and fixed active-neighbourhood depth.

### Changed

- Frontend/backend compatibility interface updated from GU26 revision 3.0 to GU28 revision 3.2.
- Packaged CUDA backend renamed from `gpu26CudaBackend` to `gpu28CudaBackend`.
- Primary solver-control dictionary renamed from `GPU2_6` to `GPU2_8`; the frontend retains a compatibility fallback for `GPU2_6`.
- `gpuResidentJammingPressure` now enables the conservative packing projection.
- Turbulence selection now supports laminar, WALE, Smagorinsky, and k-omega SST modes through standard OpenFOAM `momentumTransport` semantics.

### Removed from the governing equations

- `gpuResidentJammingOnset`
- `gpuResidentJammingPressureScale`
- `gpuResidentJammingRegularization`

The three legacy entries can remain in an older case dictionary, but GPU2.8 does not read them or apply them to the numerical equations.

### Compatibility

- The GPU2.8 frontend requires a GU28-compatible GPU2.8 backend.
- GPU2.6 and GPU2.8 backend executables are not interchangeable.
- The public frontend executable remains `GpuGkp`.
- CUDA backend implementation source remains private; the release contains the public OpenFOAM frontend source and executable-only CUDA backend.

## [2.6.0] - 2026-08-05
