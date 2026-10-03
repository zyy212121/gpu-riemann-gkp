# GPU–Riemann–GKP: gas-particle flow and layered validation

This package provides `gasUGKP` for gas and particle flow, including the
original interphase momentum and sensible-heat exchange. It excludes FSH/CHT,
solid conduction, finite-contact particle-wall heat transfer, solidification,
and radiation extensions. This standalone package can evolve independently of the engineering package;
use its local frontend and CUDA backend together.

## Shared operator implementation

The common device headers own collision-pool contributions, particle moment
contributions and publication, reduction-task operations, particle copying and
buffer commits, pressure update components, gas fluxes, and particle transport
geometry. Scalar precision, material properties, contact constraints, and
wall-energy exchange are compile-time capabilities.

The independent source review found remaining duplication in host orchestration,
directory construction, and some selection/pressure/drag adapters. The two
packages maintain mirrored common files. Shared device bodies therefore do not
establish one complete host pipeline or prove every execution strategy optimal.

Full-cell collision-pool accumulation uses named scalar accumulators. The reduction array is formed after the particle traversal, preserving contribution order, reduction order, sampling, and random-number updates. Gas, FSH, and CHT instantiate this same operator through their existing field and scalar adapters.

The particle field registry defines the payload copied during compaction and
the buffers exchanged at commit. S1 and S2 research states use the same physical particle
operations; S2 changes how heavy-cell reductions are divided among blocks.
The scheduling tile is calculated from the current block size, GPU SM count,
actual kernel residency, and directory population. Selecting S2 remains the
user's responsibility; no case-specific occupancy coefficient is used.


Tracking has a separate grid obtained from the compiled tracking kernel's
occupancy, B2, SM count, and particle capacity. Changing the reduction hierarchy
does not change this tracking grid. S1 and the single-task/multi-task L2 moment
paths use one cell publication function while retaining their reduction order.
Every nonempty cell has an explicit reduction task in the current L2 directory;
heavy cells are subdivided. A static gas directory type does not mean ordinary
cells are implicit or that the directory contains only heavy cells.

The gas volume-fraction source uses the linear discrete update. For a uniform
source without spatial gradients, the mass scaling is
`1 + dt * cepsG`; with the existing discrete volume-fraction derivative this
matches `epsG_old / epsG_new`. The source also applies the corresponding
momentum and pressure-work terms.



## Independent review and known limits (2026-10-03)

See `DEVELOPMENT_LOG_20261003_ZH.md` for the source-review scope, exact timing
protocol, and remaining maintenance work. Operator tests include tracking-grid
invariance and the three real moment publication paths. Thermal applications
also test contact detachment without leaking contact age into mechanical theta.
These checks cover local contracts; they do not certify all coupled physics.

The existing gas-particle heat update uses inconsistent finite/infinite bath
exchange amounts. The original FSH/CHT constant-Cp witness has about 1.09% scaled
closure error, and the error depends on timestep/loading. The current fixes
leave that witness unchanged. It remains a conservation defect, outside a safe
small patch; it is not reported as a passing physical validation. A robust fix
needs a shared accepted heat amount, true material enthalpy, consistent limits,
and separate gas/wall contributions for ordinary and cold-wall particles.

## CUDA operator regression checks

Run the operator checks against this checkout with CUDA available:

```bash
python3 tools/validate_shared_operators.py --output /tmp/ugkp-operator-checks
```

The command uses `CUDA_HOME` (default `/usr/local/cuda`) and
`UGKWP_CUDA_ARCH` (default `sm_89`). It compiles and launches the actual
operator implementations and
checks discrete volume-fraction source balances, collision selection and
pool totals, limited-pressure flux budgets, particle moments, indexing,
geometry, and enabled thermal payload and finite-contact ledgers. Generated
binaries and logs stay in the specified directory outside the source tree.
Use `--app gasUGKP`, `--app FSH`, or `--app CHT` to restrict the applications
when they are available in the checkout. These local checks complement
full-case validation and performance measurements.

## Build and run

Requirements: Ubuntu 22.04, OpenFOAM Foundation 10, and a CUDA toolkit and GPU.

```bash
./Allwmake
source scripts/activate.sh
gasUGKP -case examples/consistency/dustyBox
```

The default CUDA architecture is `sm_89`; set `UGKWP_CUDA_ARCH` for another
target. The root build installs into this package's `build/bin`, independently
of the shared OpenFOAM user binaries. The entry scripts fail if these local
executables are absent. Use the matching frontend/backend pair.

For stage timing and particle-validity diagnostics, build with:

```bash
UGKP_DEVELOPMENT_PROBES=1 ./Allwmake
source scripts/activate.sh
```

## Scheduling and research variants

Configure `constant/schedulingProperties`. `gpuParticleBlockThreads` controls
particle transport; `gpuReductionBlockThreads` controls cell reductions.
Supported block sizes are 32, 64, 128, and 256.

These values are **threads per CUDA block**, not warp counts or the total
number of blocks. A warp contains 32 threads: a value of 64 means two warps
per block. Both dictionary entries default to 128 when omitted.

Engineering `gpuCsrLevel` accepts:

- `L0`: direct atomic reduction without the cell-local particle directory.
- `L1`: cell-local directory, warp aggregation, and split pre-transport directory.
- `L2`: L1 plus segmented heavy-cell reductions.
- `auto`: keeps the L1 directory and periodically evaluates L2 activation.

Engineering L1 differs from the paper's L1. For controlled ablation studies,
the optional `gpuResearchVariant` selects:

| Research state | Cell-local | Warp aggregation | Split directory | Heavy reduction |
| --- | --- | --- | --- | --- |
| L0 | Off | Off | Off | Off |
| L1 | On | Off | Off | Off |
| T1 (E1) | On | On | Off | Off |
| S1 | On | On | On | Off |
| T2 (E2) | On | On | Off | On |
| S2 | On | On | On | On |

Omitting `gpuResearchVariant` retains engineering behavior. Research variants
cannot be combined with `gpuCsrLevel auto`. T2/S2 use the current heavy-cell
implementation and are not replicas of historical paper binaries.

Explicit `L2` (or research `S2`) is a user-selected execution mode. The
application does not disable it based on an estimated speedup. Once enabled,
task segmentation uses the current reduction block size, device SM count,
kernel occupancy, and directory population to determine its adaptive tile
size. This scheduling tile is not a profitability criterion for selecting L2.
`auto` remains a separate, explicitly selected mode.

## Layer names across packages

The standalone gas package exposes the research hierarchy `L0`, `L1`, `T1`,
`S1`, and `S2`. The thermal package uses `L0`, `L1`, and `L2`, corresponding
to gas research `L0`, `S1`, and `S2`. The intermediate gas research `L1` and
`T1` are not separate thermal levels. Dictionary `gpuCsrLevel L1/L2` denotes
the engineering paths; `gpuResearchVariant S1/S2` names the corresponding
research states in the standalone package. These names do not add extra
thermal levels. Explicit seven-load S2/S1 (thermal L2/L1) comparisons select
both levels directly; `auto` is a separate optional mode. Hardware block
configuration and adaptive heavy-cell tile calculation remain active when
the layer is explicitly selected.

## GPU execution and retained optimizations

L2 uses a persistent queue: resident blocks claim work until it is exhausted.
Collision-pool, moment, and finalization stages use the same
queue protocol, with a cursor reset before each consumer. The compact task
directory contains one descriptor per nonempty ordinary cell and one per
heavy-cell segment. Empty source cells create no consumer queue entries.
Workers consume these descriptors directly; ordinary-cell physics and the
cell-local particle directory are preserved.

| Optimization | Work avoided or dependency used |
| --- | --- |
| Deferred granular-temperature read | Poisson-rejected particles do not need a temperature load; RNG updates and accepted contributions are preserved. |
| Fused task setup and publication | Existing producer stages perform initialization and publish descriptor counts without separate setup passes. |
| Full/base/split compile-time specialization | The launch's known directory type removes repeated runtime source selection. |
| Pool-target and moment-recovery fusion | Final cell sums immediately feed their dependent operations; restart initialization still completes all sums. |
| Reduction-tree pruning | Only accumulation nodes contributing to the final result and valid warp levels are retained. |
| Loop-invariant thermal factors | Cell factors are reused and a disabled exchange term does not read particle temperature. |
| Bounded integer pool counts | Counts use the existing integer particle-capacity contract. |
| Heavy-cell probability reuse | Only multi-segment cells prepare a reusable current-step collision probability, after pressure preparation and state recovery. Ordinary cells calculate it where consumed. |
| Exact-survivor moment/gather fusion | In cell-local paths (including S1 and S2), the post-transport directory contains exactly the surviving particles. Their final copy is performed during moment accumulation, removing a separate gather traversal and launch; pressure projection updates the compact payload before commit. Restart initialization remains non-gathering. No empirical activation threshold is added. |
| All-live gather fast path | Existing survivor counts allow the filtering scan to be skipped when every source particle survives. All payload fields are still copied. |
| Shared moment, queue, and gather implementation | The S1/L2 particle traversal, eight-component moment reduction, and primary-field copy use common compile-time code also used by the thermal solvers. Scheduling and data movement preserve explicit physical-model and payload adapters without runtime dispatch. |

These transformations use data dependencies and supported solver settings,
without choosing policies from case names or benchmark outcomes. They preserve
the physical model; floating-point reduction ordering can affect roundoff and
stochastic trajectories. Their elapsed-time benefit depends on the workload
and GPU; this list does not imply a fixed speedup.


## Shared particle kernels

Particle tracking, collision selection and moments, Gaussian sampling,
weighted sampling correction, task construction and queues, moment recovery,
particle copying, and pressure projection are maintained in `common/`. Scalar
and physical-field adapters are selected at compile time. The thermal package
uses the same operator implementations with its own physical extensions; this
package does not instantiate finite-contact or conjugate heat-transfer models.

S1 and S2 use the same survivor directory and fused payload placement. S2 only
partitions the heavy-cell reduction work. Full and split pressure directories
consume the same limited face flux. Restart recovery does not compact particles.

## Package contents

| Path | Contents |
| --- | --- |
| `applications/gasUGKP/` | Frontend, CUDA backend, and solver tests |
| `common/`, `applications/gasUGKP/gpu/` | Shared gas, particle, and scheduling components, and the CUDA backend |
| `examples/consistency/` | Physical consistency and acoustic-convergence cases |
| `examples/paper_validation/` | Converted paper inputs preserving their discretization |
| `examples/performance/` | Nozzle cases and seven heavy-load input levels |
| `examples/research/taylorGreen/` | Short periodic-transport seed |
| `tools/` | Input migration, seed generation, and paired research runs |
| `legacy/public-3.0/`, `legacy/study-cases/` | Historical programs, inputs, and study scripts |
| `manuscript_numerical_evidence/`, `submission_addenda/` | Archived manuscript data and supplementary material |

Historical executables and results remain separate from the current solver.
Large inputs and archived result datasets may require a separate data bundle.

## Validation

```bash
source scripts/activate.sh
PYTHONPATH=tests:applications/gasUGKP/tests python3 -m pytest -q tests applications/gasUGKP/tests
examples/consistency/Allrun dustyBox sodShockTube planarCouette windSandShockTube
python3 examples/consistency/dustyBox/postprocess.py
python3 examples/consistency/sodShockTube/check_sod.py
python3 examples/consistency/planarCouette/check_couette.py
python3 examples/consistency/windSandShockTube/postprocess.py
examples/consistency/acousticWave/Allrun
```

Dusty-wave and wind-sand validation retain their dedicated constant-response-time
models. The current wind-sand input has 200 cells and 400,000 parcels; the legacy
ZIP input has 100 cells and 200,000 parcels, so they are different discretizations.

Create an independent Taylor–Green input and paired validation run (output
directories must not already exist):

```bash
python3 tools/prepare_taylor_green.py runs/taylor-input --mesh 8 --parcels-per-cell 4 --steps 20
python3 tools/run_research_matrix.py runs/taylor-input runs/taylor-check --steps 20 --repeats 1
```

The runner performs independent warmup and measurement processes, alternates
state order, checks particle balance, finite values, positivity, and final-field
differences, and records executable hashes. Full-probe short runs are validation
tools; performance studies require a separate timing protocol and sufficient
steps and repeats.

Seven heavy-load seeds are available at
`examples/performance/heavy_laval_inputs/k{1,2,4,8,16,32,64}`:

```bash
python3 tools/run_research_matrix.py examples/performance/heavy_laval_inputs/k64 runs/heavy-check \
  --states T1 S1 T2 S2 --steps 5 --repeats 1 --field-tolerance 1e-6
```

These inputs can contain millions of parcels and need several GB of GPU memory
and disk space. Keep frozen inputs and archived manuscript data when cleaning
generated run directories.

## Case migration

`tools/migrate_case.py COPIED_CASE --state S1` converts legacy `gksProperties`
or `physicalProperties` / `ugkwpProperties` inputs into the current three-dictionary
schema. It preserves physical properties and the binary particle payload while
converting restart headers. Run it on a copy; already converted cases are not
overwritten.

Legacy molecular-weight conversion uses the OpenFOAM Foundation 10 value
`RR=8314.47006650545 J/(kmol K)`. Check this conversion if your OpenFOAM
DimensionedConstants have been customized.

### Particle tracking at geometric boundaries

Particle reflection at `wedge`, `empty`, and symmetry boundaries is non-dissipative. Physical wall restitution and finite-contact thermal laws remain controlled by the wall configuration.

The maximum number of face-walk events per particle and time step is configured in `constant/schedulingProperties`:

```foam
gpuResidentMaxFaceWalkHops 512;
```

The library default is 32. Increase this limit for trajectories that cross or reflect from many faces in one time step, such as passages near a narrow wedge axis. This limit counts all face-walk events, not only physical wall impacts.


公共算子的维护所有权、字段注册、构建检查及 CUDA 调度契约见 [双库公共算子维护](docs/OPERATOR_MAINTENANCE_ZH.md)。


### 公共算子维护入口（2026-10-03 收口）

gas、FSH、CHT 的公共算子核及主机流程由 `common/` 维护。公共文件上游为 `ugkp-thermal/common`；gas 应用入口上游为独立 gas 库。修改公共实现后运行 `python3 tools/managed_mirrors.py --sync`，按登记归属同步双库；构建前检查会拒绝镜像漂移。无需分别移植三套核心实现。各应用的能力/精度适配仍需各自验证。工程结论与十对 k1/k2 结果见 [本轮结果](docs/OPERATOR_CONSOLIDATION_R2_RESULTS_ZH.md)，维护契约见 [维护说明](docs/OPERATOR_MAINTENANCE_ZH.md)。
