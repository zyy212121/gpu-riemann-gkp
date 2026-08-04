#!/usr/bin/env python3
"""Prepare a uniform 300 K gas before impulsively heating the right wall."""

from pathlib import Path


case = Path(__file__).resolve().parent
n = 64
pressure = 100000.0
gas_constant = 287.0
initial_temperature = 300.0
density = pressure / (gas_constant * initial_temperature)


def write_scalar(name: str, dimensions: str, value: float, boundaries: str) -> None:
    text = f'''FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    location "0";
    object {name};
}}
dimensions {dimensions};
internalField uniform {value:.16g};
boundaryField
{{
{boundaries}
}}
'''
    (case / "0" / name).write_text(text, encoding="utf-8")


write_scalar(
    "T",
    "[0 0 0 1 0 0 0]",
    initial_temperature,
    "    left { type fixedValue; value uniform 300; }\n"
    "    right { type fixedValue; value uniform 301; }\n"
    "    yMin { type symmetryPlane; }\n"
    "    yMax { type symmetryPlane; }\n"
    "    zEmpty { type empty; }",
)
write_scalar(
    "rho",
    "[1 -3 0 0 0 0 0]",
    density,
    "    left { type zeroGradient; }\n"
    "    right { type zeroGradient; }\n"
    "    yMin { type symmetryPlane; }\n"
    "    yMax { type symmetryPlane; }\n"
    "    zEmpty { type empty; }",
)
