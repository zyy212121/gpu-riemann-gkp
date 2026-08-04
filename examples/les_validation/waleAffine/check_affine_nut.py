#!/usr/bin/env python3
"""Compare centre-cell nut with the OpenFOAM 10 WALE affine value."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


CASE = Path(__file__).resolve().parent
C_W = 0.325
DELTA = 1.0
MATRICES = {
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


def matrix_product(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def symmetric(a):
    return tuple(
        tuple(0.5 * (a[i][j] + a[j][i]) for j in range(3))
        for i in range(3)
    )


def deviatoric(a):
    trace_third = sum(a[i][i] for i in range(3)) / 3.0
    return tuple(
        tuple(a[i][j] - (trace_third if i == j else 0.0) for j in range(3))
        for i in range(3)
    )


def magnitude_squared(a):
    return sum(value * value for row in a for value in row)


def openfoam_wale_nut(gradient):
    # OpenFOAM 10 WALE.C:
    # Sd=dev(symm(gradU & gradU)); the first denominator invariant is
    # magSqr(symm(gradU)), not magSqr(dev(symm(gradU))).
    symmetric_gradient_squared = magnitude_squared(symmetric(gradient))
    sd_squared = magnitude_squared(
        deviatoric(symmetric(matrix_product(gradient, gradient)))
    )
    numerator = sd_squared ** 1.5
    denominator = (
        symmetric_gradient_squared ** 2.5 + sd_squared ** 1.25
    )
    return C_W * C_W * DELTA * DELTA * numerator / denominator


def latest_time() -> Path:
    times = []
    for path in CASE.iterdir():
        try:
            value = float(path.name)
        except ValueError:
            continue
        if path.is_dir() and value > 0.0:
            times.append((value, path))
    if not times:
        raise RuntimeError("no result time directory was written")
    return max(times)[1]


def scalar_values(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if match:
        values = [float(value) for value in match.group(2).split()]
        if len(values) != int(match.group(1)):
            raise RuntimeError("invalid nut list length")
        return values
    match = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if not match:
        raise RuntimeError("cannot parse nut internalField")
    return [float(match.group(1))] * 27


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=tuple(MATRICES), required=True)
parser.add_argument("--output", type=Path, required=True)
arguments = parser.parse_args()

time_dir = latest_time()
nut = scalar_values(time_dir / "nut")
if len(nut) != 27:
    raise RuntimeError("expected 27 cells")
centre = nut[13]
expected = openfoam_wale_nut(MATRICES[arguments.mode])
relative_error = abs(centre - expected) / expected
if not math.isfinite(centre) or centre < 0.0:
    raise RuntimeError("central nut is non-finite or negative")
if relative_error > 1.0e-6:
    raise RuntimeError(
        f"WALE affine check failed for {arguments.mode}: "
        f"value={centre:.16g}, expected={expected:.16g}, "
        f"relativeError={relative_error:.3e}"
    )

report = {
    "time": time_dir.name,
    "model": "WALE",
    "mode": arguments.mode,
    "velocityGradient": MATRICES[arguments.mode],
    "divergence": sum(MATRICES[arguments.mode][i][i] for i in range(3)),
    "centralCell": 13,
    "nut": centre,
    "expectedNut": expected,
    "relativeError": relative_error,
}
output = arguments.output
if not output.is_absolute():
    output = CASE / output
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
