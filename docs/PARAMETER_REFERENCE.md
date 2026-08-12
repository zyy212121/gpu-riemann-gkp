# GPU-Riemann-GKP 3.0 Parameter Reference

The release follows OpenFOAM dictionary and field semantics.  Gas numerical
controls live in `system/fvSchemes` and `system/fvSolution`; particle and GPU
execution controls live in `constant/ugkwpProperties`; gravity uses
`constant/g`.

## Gas numerics

| Location and entry | Values | Meaning |
|---|---|---|
| `fvSchemes/fluxScheme` | Tadmor/Rusanov, Kurganov/HLL, HLLE, HLLC, Roe, HLLEM, HLLC-ADC, SLAU2, SLAU2.2 | Inviscid face flux |
| `fvSchemes/ddtSchemes/default` | Euler, SSPRK2, SSPRK3 | Explicit gas time integrator |
| `fvSchemes/divSchemes` | OpenFOAM upwind/limited-linear entries | Selects first-order, MUSCL, or energy-limited reconstruction |
| `fvSolution/GpuGkp/rhoMin` | positive scalar | Gas-density floor |
| `fvSolution/GpuGkp/TMin` | positive scalar | Gas-temperature floor |
| `fvSolution/GpuGkp/robustFallback` | true/false | Positivity fallback for supported low-dissipation fluxes |
| `fvSolution/GpuGkp/maxDiffusionNumber` | positive scalar | Viscous/thermal explicit-step limit |

OpenFOAM `controlDict` entries `adjustTimeStep`, `maxCo`, `maxDeltaT`,
`writeControl`, and `writeInterval` retain their standard meanings.

## Turbulence

`constant/momentumTransport` selects `laminar`, LES `WALE`, LES
`Smagorinsky`, or RAS `kOmegaSST`.  SST supports `wallTreatment lowRe` and
`wallTreatment wallFunction`, plus `kMin`, `omegaMin`, `maxSourceNumber`, and
the standard coefficient set carried by the model dictionary.

## Particle model and capacity

| Entry | Default/constraint | Meaning |
|---|---|---|
| `gpuResidentPureGasOnly` | false | Omits the particle execution path when true |
| `gpuResidentParticleCapacity` | non-negative integer | Maximum resident parcel count |
| `gpuResidentRandomSeed` | integer | Reproducible GPU random seed |
| `gpuResidentMaxFaceWalkHops` | positive integer | Maximum face crossings per tracking call |
| `gpuResidentCourantUpdateInterval` | positive integer | GPU steps between complete Courant updates |
| `gpuResidentMaxDeltaTGrowth` | positive scalar | Maximum ratio between consecutive adaptive steps |
| `rhoS` | positive scalar | Particle material density |
| `dS` | positive scalar | Representative/fixed diameter |
| `dMin`, `dMax` | positive, ordered | Diameter truncation limits |
| `dSigma` | non-negative | Lognormal diameter width; zero selects fixed diameter |
| `parcelMass` | positive | Compatibility default for both explicit mass controls |
| `injectionParcelMass` | `parcelMass` | Statistical mass of newly injected parcels |
| `legacyRestartParcelMass` | `parcelMass` | Statistical mass assigned to V4/V5 restart parcels |
| `particleTemperatureTransport` | false | Particle material-temperature transport switch |
| `particleRho`, `particleCp` | positive when thermal transport is active | Particle heat-capacity properties |
| `particleTMin`, `particleTMax` | ordered | Admissible particle-temperature interval |
| `gpuResidentInjectionTheta` | non-negative | Injected fluctuating energy |
| `gpuResidentInjectionTp` | within temperature limits | Injected material temperature |

## Drag and gravity

| Entry | Values | Meaning |
|---|---|---|
| `dragModel` | `SchillerNaumann`, `GidaspowErgunWenYu` | Host-selected, CUDA-statically-specialised drag law |
| `GidaspowErgunWenYuCoeffs/residualRe` | positive | Dense-model Reynolds regularisation |
| `constant/g/value` | finite vector with acceleration dimensions | Gas and particle gravitational acceleration |

An absent `constant/g` selects zero gravity.

## Particle pressure and packing

| Entry | Meaning |
|---|---|
| `gpuResidentCollisionalPressure` | Enables the granular collisional-pressure model |
| `gpuResidentCollisionalRestitution` | Restitution coefficient used by collisional pressure |
| `gpuResidentPressureKickFraction` | Fractional particle-pressure kick substep |
| `gpuResidentJammingPressure` | Enables conservative mobile-particle packing projection |
| `gpuResidentPackingFraction` | Maximum mobile-particle volume fraction |
| `gpuResidentPackingProjectionIterations` | Jacobi iteration count and active-neighbourhood depth |
| `epsSMin`, `thetaMin` | Solid-volume and fluctuating-energy floors |

The former soft-pressure controls `gpuResidentJammingOnset`,
`gpuResidentJammingPressureScale`, and `gpuResidentJammingRegularization` are
outside the 3.0 governing equations.

## CP-CST and launch geometry

| Entry | Meaning |
|---|---|
| `gpuCsrCellLocalPath` | Enables L1 cell-local CP-CST segmentation and compaction |
| `gpuCsrWarpAggregatedBinning` | Adds E1 warp-aggregated cell counting and slot reservation |
| `gpuCsrHeavyReduction` | Adds E2 occupancy-derived heavy-cell scheduling |
| `bn` | Block exponent 5--8; CUDA block size is `B=2^bn` |

E1 and E2 require `gpuCsrCellLocalPath true`; E2 production settings normally
also enable warp aggregation.  The 2.8 fixed threshold, tile-size, and worker
count entries are retired.

## Particle boundaries and dynamic injection

`particleWallCoeffs` maps patch names to normal-velocity restitution
coefficients.  Dynamic injection uses `gpuResidentDynamicInlet`,
`gpuResidentInletPatch`, `gpuResidentPressureTable`,
`gpuResidentVolumeFractionTable`, and `gpuResidentInletTemperature`.

## Validation-backend-only entries

DustyWave and the wind--sand shock tube select the packaged
`gpu30CudaBackend-twophaseflux-validation` binary through `Allwmake` and
restore the production binary through `Allrestore`.  Their legacy-named
entries `gpuResidentDragModel`, `gpuResidentParticleResponseTime`,
`gpuResidentHeatTransferModel`, `gpuResidentParticleThermalResponseTime`, and
`gpuResidentGasHeatCapacity` are consumed by the case wrapper to define the
analytic constant-response benchmark.  Production cases use `dragModel` and
the physical Ranz--Marshall thermal implementation.

## Restart and offline refinement

GPU 3.0 writes `GPU3_PARTICLES_V1_BIN`, including per-parcel `pm`.  The tools
`tools/gpu3_particle_restart.py` and `tools/refine_particle_cells.py` inspect
and conservatively refine a stopped restart.  Refinement preserves per-cell
and global mass, momentum, translational/random energy, mass-weighted diameter,
and mass-weighted temperature.
