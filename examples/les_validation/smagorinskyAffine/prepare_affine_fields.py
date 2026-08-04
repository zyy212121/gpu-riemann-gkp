#!/usr/bin/env python3
"""Write U=Gx for a 3x3x3 block with cell centres at -1, 0 and 1 m."""

from pathlib import Path


case = Path(__file__).resolve().parent
coordinates = (-1.0, 0.0, 1.0)
velocity = []
for z in coordinates:
    for y in coordinates:
        for x in coordinates:
            velocity.append((x + 2.0 * y, -x + z, -2.0 * y - z))

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
