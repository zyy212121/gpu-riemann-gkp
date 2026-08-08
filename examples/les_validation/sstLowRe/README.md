# Explicit GPU k-omega SST smoke test

This 64-cell wall-driven channel is a short implementation smoke test for
the GPU2.8 RAS path.  It deliberately uses standard OpenFOAM files:

- `constant/physicalProperties`
- `constant/momentumTransport` with `simulationType RAS` and `kOmegaSST`
- `0/k`, `0/omega`, and the ordinary gas fields
- `fvSchemes` selecting an explicit SSPRK2 time scheme

The test is not a turbulence-validation data set.  `./Allrun` requires an
NVIDIA GPU, runs every SST stage on that GPU, and verifies finite positive
`k`, `omega`, and `nut` restart fields.

After loading the OpenFOAM-10 environment, `./check_sst_restart.py` compares
a continuous run with a split restart at `2e-5 s` and writes
`check_sst_restart.json`.
