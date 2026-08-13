# Manuscript numerical result evidence

## Scope

This directory is the result-data submission for the main-text numerical experiments in `manuscript.tex`.

It contains the figures and tables used in the manuscript, the submitted plotting CSV data, and the minimal original solver outputs from which these CSV files were extracted.

The included main-text cases are:

1. DustyBox homogeneous drag relaxation;
2. wind--sand shock tube;
3. Taylor--Green vortex consistency and performance tests;
4. two-phase Laval nozzle field-distribution and performance tests;
5. heavy-cell Laval nozzle thread-block-size and conservative local-refinement tests.

## Traceability levels

The submitted evidence is organized into three levels:

1. `figures/` contains the image files actually referenced by the manuscript;
2. `data/`, `consistency/`, `performance/`, `block_scan/`, `weighted_tail_s1_s2/`, and `four_state_extension/` contain the tables used for plotting and for reporting numerical values in the manuscript;
3. `source_calculation_outputs/`, `raw_runs*/`, `raw_physics_gate/`, and `refinement_source/` contain the original calculation outputs actually read when producing the tables above.

`FIGURE_DATA_TRACEABILITY.csv` maps every main-text figure to its corresponding plotting data.

`TABLE_DATA_TRACEABILITY.csv` performs the same traceability audit for all seven numerical tables in the main text.

`SOURCE_CALCULATION_TRACEABILITY.csv` further maps each plotting or summary data product to the type of original solver output actually read.

`DATA_INDEX.csv` is a compact file index, and `SHA256SUMS.txt` provides a byte-level checksum for every submitted file.

Historical absolute workspace paths in textual metadata are normalized to the neutral archive label `SOURCE_STUDY_ARCHIVE`. Numerical fields, particle data, timing values, validation values, and file-content hashes are unchanged by this path normalization.

## Directory layout

| Directory | Submitted evidence |
|---|---|
| `01_dustybox/` | Manuscript figure and validation tables; gas- and particle-phase velocity fields at 31 original output times; original calculation log, checker result, and case parameters. |
| `02_wind_sand_shock_tube/` | Four manuscript figures and validation tables; complete Eulerian fields at the final time; the complete 200,000-particle restart used for covariance extraction; published reference profiles, calculation log, and checker result. |
| `03_taylor_green/` | Consistency and performance figures and tables; 21 matched output times for four states; the seven classes of Eulerian fields actually read by the consistency reduction; exact particle-count header records; one complete initial particle restart shared by the four states; and all per-process CUDA timing outputs from the ten-load performance matrix. |
| `04_two_phase_laval_nozzle/` | Sparse/dense field-distribution figures and readable Eulerian field snapshots; original nozzle trajectory fields used to produce the cell CSV; original occupancy particle restart and mesh `owner` file; and all per-process CUDA timing outputs from the six-load matrix. |
| `05_heavy_laval_nozzle/` | Thread-block-size and local-heavy-load figures and tables; all per-process CUDA timing outputs and experiment receipts; source tables for seven levels of conservative local refinement; and original one-step physical-gate fields and probes for the S1/S2 and four-state matrices. |
| `provenance/` | Frozen consistency-evidence manifest and its checksum snapshot. |

## Meaning of “original calculation output”

For consistency tests, “original calculation output” means the original OpenFOAM field files and particle outputs actually parsed by the reduction program.

For performance tests, “original calculation output” means the original `stage_timing.csv` from every accepted process, together with the corresponding run manifest, solver log, validation receipt, and binary/configuration hashes. The performance conclusions in the manuscript use CUDA-event timings rather than external wall-clock timings.

For local-refinement tests, it also includes the per-parent-particle conservation table, selected-cell table, and refinement manifest for each refinement factor.

For physical-gate validation, it includes the original particle-population probes, solver logs, source-residual files, and the macro fields used in the comparison.

Taylor--Green consistency is the only data set for which the particle records are minimized by byte range. The reduction program reads all seven classes of Eulerian fields, but reads only the first line of each particle restart to obtain the particle count. This directory therefore contains all 84 exact first-line records, named `gpuResidentStrictParticles.header`, and also contains one complete initial particle restart shared by the four states. The consistency audit records that this initial restart is identical among L0, L1, T1, and S1.

The field snapshots under `04_two_phase_laval_nozzle/field_snapshots/` contain the mesh, a minimal `controlDict`, and an empty `result.foam` marker, and can therefore be opened directly with ParaView or OpenFOAM readers.

## Data conventions

- CSV files use UTF-8 encoding and contain a header row.
- JSON files preserve the original validation or experiment metadata.
- Unless otherwise stated by the column name, timing data are CUDA-event timings in milliseconds.
- A speedup named `B_over_A` is defined as `time(A) / time(B)`; a value greater than 1 means that method B is faster.
- The manuscript method names T1 and T2 correspond to the archived internal names E1 and E2, respectively. Original files retain E1/E2, whereas manuscript-facing data use T1/T2.
- All reported confidence intervals are the paired 95% confidence intervals archived by the original experiments.

## Source-data chains for individual cases

### DustyBox

`dustybox_validation.csv` was reduced from the 31 pairs of original `U` and `Us` field files under `source_calculation_outputs/`.

The relaxation history is determined by `case_parameters.json`; `log.gpu2.6` and `check_dustybox.json` preserve the calculation and acceptance evidence.

In addition to the manuscript data table, the original reduction output is retained as `source_calculation_outputs/validation_data.csv`.

### Wind--sand shock tube

`windsand_shocktube_validation.csv` was obtained from the original final-time `rho`, `epsilonS`, `U`, `Us`, `p`, and `Tp` fields, together with the published reference profiles of Yang et al.

`windsand_thermal_covariance.csv` was calculated directly from the submitted complete `gpuResidentStrictParticles.dat` file and cross-checked against the final particle-temperature field.

The original shock-tube reduction result is retained as `source_calculation_outputs/validation_data.csv`.

### Taylor--Green vortex

The consistency tables come from 21 matched output times for each of L0, L1, T1, and S1.

The original fields submitted at every output time are `rho`, `p`, `T`, `U`, `epsilonS`, `Us`, and `theta`.

The performance tables are traceable to the per-process `stage_timing.csv` directories for the low-load, high-load, and paired T1/S1 experiments.

`consistency/source_calculation_outputs/reducer_outputs/` preserves the original reduction CSV/JSON products before the archived state name E1 was changed to the manuscript state name T1.

### Two-phase Laval nozzle

`nozzle_field_t0p115.csv` was parsed from the original `p`, `epsilonS`, and `U` fields in the nozzle trajectory.

The occupancy statistics come from the submitted original base particle restart and mesh `owner` file.

The load-dependent performance tables are traceable to all 144 per-process `stage_timing.csv` files and their run receipts.

The sparse/dense distribution figures are traceable to the submitted readable original Eulerian result snapshots.

### Heavy-cell Laval nozzle

The thread-block-size scan is traceable to all 52 archived process timing files.

The S1/S2 heavy-tail matrix is traceable to 84 process timing files, and the T1/T2/S1/S2 extension matrix is traceable to 168 process timing files.

Every local-refinement level contains its original conservation and cell-selection records.

The submitted original one-step physical gates contain the particle-population ledger and macro fields used to confirm physical consistency before the performance comparisons.

## Integrity

`SHA256SUMS.txt` covers every submitted file except itself. Paths in the checksum file are relative to this directory and use forward slashes.

`provenance/cross_case_performance_source.csv` preserves the combined Taylor--Green/nozzle intermediate table. The case-specific and cross-case comparison subsets used in the manuscript were selected from this table; its upstream data are the per-process original timing trees submitted in this directory.
