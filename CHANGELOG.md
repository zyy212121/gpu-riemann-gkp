# Changelog

All notable user-facing changes to GPU-Riemann-GKP are recorded here.

## Unreleased

No changes after the 3.0.0 release.

## [3.0.0] - 2026-08-12

### Added

- Restart-persistent per-parcel statistical mass in the new
  `GPU3_PARTICLES_V1_BIN` chunked binary format.
- Separate `injectionParcelMass` and `legacyRestartParcelMass` controls while
  retaining `parcelMass` as their compatibility default.
- Conservative stopped-state particle refinement tools for selected cells.
- Optional standard OpenFOAM `constant/g` gravity input.
- Runtime drag selection between `SchillerNaumann` and
  `GidaspowErgunWenYu`, with CUDA kernels statically specialised per model.
- Dynamic particle block-size control `bn`, accepting 5--8 for 32--256
  threads per block.

### Changed

- Frontend/backend protocol advanced to `G300` revision 5.0.
- Packaged backend renamed to `gpu30CudaBackend` and rebuilt for
  `sm_75`, `sm_80`, `sm_86`, `sm_89`, and `sm_90`, with `compute_90` PTX.
- Unequal-weight collision correction now conserves mass-weighted momentum
  and random energy while retaining each parcel's statistical mass.
- Heavy-cell launch geometry is derived from `bn` and device occupancy; the
  retired fixed heavy threshold/tile/worker controls are no longer read.
- Renamed the canonical `system/fvSolution` solver-control dictionary to the
  release-neutral product name `GpuGkp`.

### Compatibility

- GPU2 V4/V5 particle restarts remain readable; their missing per-parcel mass
  is supplied by `legacyRestartParcelMass`.
- GPU 3.0 writes `GPU3_PARTICLES_V1_BIN`; older solvers do not read it.
- `GPU2_8` and `GPU2_6` remain warning-producing `fvSolution` migration
  aliases; defining more than one canonical/legacy block is rejected.
- The public frontend executable remains `GpuGkp`.
- GPU 3.0 frontends and older backend executables are not interchangeable.
- CUDA implementation source remains private; the release contains the
  public OpenFOAM frontend and executable-only CUDA backends.

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
