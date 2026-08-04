#!/usr/bin/env python3
"""Write U=Gx for trace-free or compressible WALE affine checks."""

import argparse
from pathlib import Path


case = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=("trace_free", "compressible"),
    default="trace_free",
)
mode = parser.parse_args().mode

# The original matrix G0 is divergence-free.  G0 + 2I retains the same
# deviatoric part but has div(U)=6 1/s, exposing the compressible WALE
# denominator invariant used by OpenFOAM 10.
matrices = {
    "trace_free": (
        (1.0, 2.0, 0.0),
        (-1.0, 0.0, 1.0),
        (0.0, -2.0, -1.0),
    ),
    "compressible": (
        (3.0, 2.0, 0.0),
        (-1.0, 2.0, 1.0),
        (0.0, -2.0, 1.0),
    ),
}
gradient = matrices[mode]
coordinates = (-1.0, 0.0, 1.0)
velocity = []
for z in coordinates:
    for y in coordinates:
        for x in coordinates:
            position = (x, y, z)
            velocity.append(
                tuple(
                    sum(gradient[i][j] * position[j] for j in range(3))
                    for i in range(3)
                )
            )

body = "\n".join(f"    ({u:.16g} {v:.16g} {w:.16g})" for u, v, w in velocity)
text = f'''FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    location "0";
    object U;
}}
dimensions [0 1 -1 0 0 0 0];
internalField nonuniform List<vector>
27
(
{body}
);
boundaryField
{{
    xMin {{ type fixedValue; value uniform (0 0 0); }}
    xMax {{ type fixedValue; value uniform (0 0 0); }}
    yMin {{ type fixedValue; value uniform (0 0 0); }}
    yMax {{ type fixedValue; value uniform (0 0 0); }}
    zMin {{ type fixedValue; value uniform (0 0 0); }}
    zMax {{ type fixedValue; value uniform (0 0 0); }}
}}
'''
(case / "0" / "U").write_text(text, encoding="utf-8")
print(f"preparedMode={mode}")
