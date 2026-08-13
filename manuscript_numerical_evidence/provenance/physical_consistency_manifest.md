# Physical-consistency figure manifest

## Provenance

The authoritative archive is
`/home/lss/OpenFOAM/lss-10/applications/solvers/SOURCE_STUDY_ARCHIVE/cases/consistency`.
Its frozen snapshot is indexed by
`SOURCE_STUDY_ARCHIVE/manifests/CONSISTENCY_SNAPSHOT.sha256`. The archived runs
were produced by the GPU2.6 physical-validation workflow before the
particle--cell execution variants were introduced. They validate the inherited
two-phase physical core and do not constitute L0/L1/T1/T2/S1/S2 performance
measurements.

The complete OpenFOAM time directories and the 200,000-particle restart remain
in the authoritative archive. This directory contains the compact publication
data, acceptance metrics, plotting programs and output figures.

## Exported evidence

| SHA-256 | File | Role |
|---|---|---|
| `7a88b6820a4c2377f4af1462b6131b7eadeee63742b0bf7b5ab27f1ed831d3e9` | `dustybox_validation.csv` | 31 analytical/numerical relaxation samples and error histories |
| `fafc84441c84fee6279f39401409a52260bd400a59dac3ddab320511ac18ce1d` | `dustybox_validation_metrics.json` | Frozen DustyBox acceptance metrics |
| `fefb36170cc15fb8f60218d5bac5d2db5ef6825584eda05e7b376b20418a675e` | `dustybox_case_parameters.json` | DustyBox setup |
| `a7dd2b37338a46fcccd8554ffabfd5dad8026ecf05b594c578df967d5f2df7f3` | `windsand_shocktube_validation.csv` | 100-cell published and present profiles |
| `8e9ab3319fb57d555ec9542a0d22ab2c1e1e729016c282f1578c62e10986456d` | `windsand_shocktube_validation_metrics.json` | Frozen profile, positivity, mass and stored-energy diagnostics |
| `5fa4862fd79b87c49c91e5a67bcd3e429c1360c66c7aa6f8678c9d4fada8dba2` | `windsand_case_parameters.json` | Wind--sand shock-tube setup |
| `29e54bf417d509b578be82047c1d97909ea939393cfd86765b05b712f2bb0719` | `windsand_thermal_covariance.csv` | Per-cell covariance regenerated from the frozen particle restart |
| `5b4f985f17a50c005fa55050bd52e5a4c761207f18af927b626b7df1a89d08dc` | `source_data/windsand_yang2022/yang2022_collisionless_ugkwp.csv` | Reference curves extracted from the published vector plots |

The four source PDFs and `extract_profiles.py` in
`source_data/windsand_yang2022` preserve the vector-curve extraction route for
the Yang et al. (2022) reference data (DOI
`10.4208/cicp.OA-2021-0153`).

The generated publication figures are `dustybox_validation.{png,pdf}`;
`windsand_shocktube_density.{png,pdf}`,
`windsand_shocktube_velocity.{png,pdf}` and
`windsand_shocktube_pressure_temperature.{png,pdf}`; and
`windsand_thermal_covariance.{png,pdf}`. The six wind--sand fields are split
into three one-by-two assets for legibility; the manuscript presents them as
three consecutive two-panel layout blocks, panels (a)--(f), with one shared caption after the final block.

## Regeneration

From this directory, regenerate the covariance CSV with the archived restart:

```powershell
python windsand_thermal_covariance_extract.py `
  --particle-file "\\wsl.localhost\Ubuntu-22.04\home\lss\OpenFOAM\lss-10\applications\solvers\SOURCE_STUDY_ARCHIVE\cases\consistency\windSandShockTube\0.2\gpuResidentStrictParticles.dat"
```

Regenerate the English PNG/PDF figure pairs with:

```powershell
python dustybox_validation.py
python windsand_shocktube_validation.py
python windsand_thermal_covariance.py
```

## Claim limits

- DustyBox supports homogeneous nonlinear drag relaxation, spatial uniformity
  and mixture-momentum consistency within the recorded tolerances.
- The wind--sand shock tube supports six-field profile agreement, positivity
  and mass consistency against Yang et al. (2022).
- The wind--sand stored-energy diagnostic changes by approximately `-6.31%`.
  This case must not be cited as evidence of strict total-energy conservation.
