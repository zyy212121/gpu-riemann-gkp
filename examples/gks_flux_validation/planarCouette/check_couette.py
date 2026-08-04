#!/usr/bin/env python3
"""Compare transient impulsive Couette flow with its Fourier-series solution."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


CASE = Path(__file__).resolve().parent
N = 64
MU = 0.1
RHO = 1.0
WALL_SPEED = 1.0


def latest_time() -> tuple[float, Path]:
    times: list[tuple[float, Path]] = []
    for path in CASE.iterdir():
        try:
            value = float(path.name)
        except ValueError:
            continue
        if path.is_dir() and value > 0.0:
            times.append((value, path))
    if not times:
        raise RuntimeError("no result time directory was written")
    return max(times)


def vectors(path: Path) -> list[tuple[float, float, float]]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if match:
        return [
            tuple(float(v) for v in item.split())
            for item in re.findall(r"\(([^()]*)\)", match.group(2))
        ]
    match = re.search(r"internalField\s+uniform\s+\(([^()]*)\)\s*;", text)
    if not match:
        raise RuntimeError(f"cannot parse vector field {path}")
    value = tuple(float(v) for v in match.group(1).split())
    return [value] * N


def scalars(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if match:
        return [float(v) for v in match.group(2).split()]
    match = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if not match:
        raise RuntimeError(f"cannot parse scalar field {path}")
    return [float(match.group(1))] * N


def analytic_velocity(y: float, time_value: float) -> float:
    nu = MU / RHO
    value = y
    for mode in range(1, 401):
        value += (
            2.0
            * ((-1.0) ** mode)
            * math.sin(mode * math.pi * y)
            * math.exp(-(mode * math.pi) ** 2 * nu * time_value)
            / (mode * math.pi)
        )
    return WALL_SPEED * value


time_value, time_dir = latest_time()
velocity = vectors(time_dir / "U")
nut = scalars(time_dir / "nut")
if len(velocity) != N or len(nut) != N:
    raise RuntimeError("unexpected cell count")
expected = [analytic_velocity((j + 0.5) / N, time_value) for j in range(N)]
errors = [abs(value[0] - target) for value, target in zip(velocity, expected)]
l2_error = math.sqrt(sum(error * error for error in errors) / N)
cross_velocity = max(max(abs(value[1]), abs(value[2])) for value in velocity)
if not all(math.isfinite(component) for value in velocity for component in value):
    raise RuntimeError("non-finite velocity")
if l2_error > 4.0e-2 or max(errors) > 1.2e-1 or cross_velocity > 2.0e-4:
    raise RuntimeError("transient Couette diffusion does not match the analytic solution")
if max(abs(value) for value in nut) > 1.0e-14:
    raise RuntimeError("laminar mode produced non-zero nut")
if max(value[0] for value in velocity) < 5.0e-2:
    raise RuntimeError("moving-wall momentum did not diffuse into the gas")

report = {
    "time": time_dir.name,
    "kinematicViscosity": MU / RHO,
    "l2UxError": l2_error,
    "maxUxError": max(errors),
    "maxCrossVelocity": cross_velocity,
    "maxAbsNut": max(abs(value) for value in nut),
}
(CASE / "check_couette.json").write_text(json.dumps(report, indent=2) + "\n")
with (CASE / "validation_data.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(("y_m", "velocity_exact_m_s", "velocity_numerical_m_s"))
    for index, (target, value) in enumerate(zip(expected, velocity)):
        writer.writerow(((index + 0.5) / N, target, value[0]))
print(json.dumps(report, indent=2))
