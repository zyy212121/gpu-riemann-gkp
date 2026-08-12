# Upgrade from GPU-Riemann-GKP 2.8 to 3.0

GPU-Riemann-GKP 3.0 keeps the public OpenFOAM executable name `GpuGkp` and the
standard case layout.  The release advances the separated CUDA protocol,
introduces persistent parcel weights, adds configurable gravity and drag, and
replaces fixed heavy-cell launch parameters with dynamic block geometry.

## Required actions

1. Run `./install.sh` from the 3.0 tree.  This installs
   `gpu30CudaBackend` and rebuilds the `GpuGkp` frontend.
2. Keep `application GpuGkp;` in `system/controlDict`.
3. Use the canonical `GpuGkp` dictionary in `system/fvSolution`.
4. Remove the retired entries
   `gpuCsrHeavyCellThreshold`, `gpuCsrHeavyTileParticles`, and
   `gpuCsrHeavyWorkerBlocksPerSM` from `constant/ugkwpProperties`.
5. Add `bn 8;` for the former 256-thread production configuration.  Valid
   values are 5, 6, 7, and 8.
6. For particle cases, declare the restart and injection mass semantics:

   ```foam
   parcelMass                 5e-9;
   injectionParcelMass        5e-9;
   legacyRestartParcelMass    5e-9;
   ```

   Equal values reproduce a uniform-weight 2.8 case.  `parcelMass` supplies
   the default when either explicit entry is omitted.

## Restart compatibility

GPU 3.0 reads the legacy `GPU2_PARTICLES_V4` and `GPU2_PARTICLES_V5_BIN`
formats.  Since those formats contain no parcel-mass array, every imported
representative receives `legacyRestartParcelMass`.

GPU 3.0 writes `GPU3_PARTICLES_V1_BIN`.  The new format stores the statistical
mass of every parcel and preserves heterogeneous weights through write,
restart, compaction, and collision sampling.  A 3.0 particle file requires a
3.0 frontend/backend pair.

## Drag and gravity

The default drag model remains Schiller--Naumann:

```foam
dragModel SchillerNaumann;
```

The dense-bed alternative is:

```foam
dragModel GidaspowErgunWenYu;

GidaspowErgunWenYuCoeffs
{
    residualRe 1e-3;
}
```

Gravity uses the optional standard OpenFOAM `constant/g` field.  When the file
is absent, the resolved acceleration is exactly `(0 0 0)` and the gravity
kernel is omitted.

## CP-CST levels

| Level | `gpuCsrCellLocalPath` | `gpuCsrWarpAggregatedBinning` | `gpuCsrHeavyReduction` |
|---|---:|---:|---:|
| L0 | false | false | false |
| L1 | true | false | false |
| E1 | true | true | false |
| E2 | true | true | true |

`bn` applies to all four paths.  The E2 heavy schedule derives its tile and
resident worker geometry from `B=2^bn` and CUDA occupancy.

## Backend compatibility

The 3.0 process protocol is `G300` revision 5.0.  Set a custom backend path
with `GPU30_CUDA_BACKEND`.  GPU 2.8 and GPU 3.0 backend executables cannot be
mixed with the other release's frontend.

## Example migration status

Every packaged example retains its 2.8 physical setup and minimum runnable
state.  Particle examples explicitly define mass semantics and `bn`; pure-gas
examples retain zero particle loading.  DustyWave and the wind--sand shock
tube use the supplied 3.0 constant-response validation backend through their
existing `Allwmake`/`Allrestore` workflow.
