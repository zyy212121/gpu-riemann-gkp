# Explicit GPU SST wall-function smoke test

This 64-cell channel selects `wallTreatment wallFunction`.  The initial gas
has an axial velocity of `1 m/s` and temperature `1200 K`; both no-slip walls
are held at `750 K`.  The test verifies that the standard SST restart fields
remain finite, the startup log confirms the resident CUDA wall-function
path, and the conservative gas energy responds to the GPU wall heat flux.

Run with `./Allrun` on an NVIDIA GPU.
After loading the OpenFOAM environment,
`./check_wall_function_restart.py` verifies a split restart.
