#!/usr/bin/env python3
"""Write a low-amplitude standing acoustic eigenmode into one closed tube."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


GAMMA = 1.4
R_GAS = 287.0
P0 = 100000.0
T0 = 300.0
RHO0 = P0 / (R_GAS * T0)
SOUND = math.sqrt(GAMMA * P0 / RHO0)
AMPLITUDE = 1.0
PHASE = math.pi / 4.0
WAVENUMBER = math.pi


def scalar_field(case: Path, name: str, dimensions: str, values: list[float]) -> None:
    body = "\n".join(f"{value:.17g}" for value in values)
    text = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    location \"0\";
    object {name};
}}
dimensions {dimensions};
internalField nonuniform List<scalar>
{len(values)}
(
{body}
)
;
boundaryField
{{
    left {{ type zeroGradient; }}
    right {{ type zeroGradient; }}
    yEmpty {{ type empty; }}
    zEmpty {{ type empty; }}
}}
"""
    (case / "0" / name).write_text(text)


def vector_field(case: Path, values: list[float]) -> None:
    body = "\n".join(f"({value:.17g} 0 0)" for value in values)
    text = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    location \"0\";
    object U;
}}
dimensions [0 1 -1 0 0 0 0];
internalField nonuniform List<vector>
{len(values)}
(
{body}
)
;
boundaryField
{{
    left {{ type fixedValue; value uniform (0 0 0); }}
    right {{ type fixedValue; value uniform (0 0 0); }}
    yEmpty {{ type empty; }}
    zEmpty {{ type empty; }}
}}
"""
    (case / "0" / "U").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("cells", type=int)
    args = parser.parse_args()
    if args.cells <= 0:
        raise SystemExit("cells must be positive")

    dx = 1.0 / args.cells
    average_factor = math.sin(0.5 * WAVENUMBER * dx) / (0.5 * WAVENUMBER * dx)
    pressure: list[float] = []
    density: list[float] = []
    temperature: list[float] = []
    velocity: list[float] = []
    for cell in range(args.cells):
        x = (cell + 0.5) * dx
        p_prime = (
            AMPLITUDE * average_factor * math.cos(WAVENUMBER * x)
            * math.cos(PHASE)
        )
        u = (
            AMPLITUDE / (RHO0 * SOUND) * average_factor
            * math.sin(WAVENUMBER * x) * math.sin(PHASE)
        )
        p = P0 + p_prime
        rho = RHO0 + p_prime / (SOUND * SOUND)
        pressure.append(p)
        density.append(rho)
        temperature.append(p / (rho * R_GAS))
        velocity.append(u)

    scalar_field(args.case, "p", "[1 -1 -2 0 0 0 0]", pressure)
    scalar_field(args.case, "rho", "[1 -3 0 0 0 0 0]", density)
    scalar_field(args.case, "T", "[0 0 0 1 0 0 0]", temperature)
    vector_field(args.case, velocity)


if __name__ == "__main__":
    main()
