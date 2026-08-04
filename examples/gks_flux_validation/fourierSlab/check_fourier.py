#!/usr/bin/env python3
"""Compare transient wall heating with the one-dimensional Fourier solution."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


CASE = Path(__file__).resolve().parent
N = 64
T_LEFT = 300.0
T_RIGHT = 301.0
PRESSURE = 100000.0
R_GAS = 287.0
GAMMA = 1.4
MU = 0.02
PR = 0.72


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


CP = GAMMA * R_GAS / (GAMMA - 1.0)
RHO_REFERENCE = PRESSURE / (R_GAS * T_LEFT)
CONDUCTIVITY = MU * CP / PR
THERMAL_DIFFUSIVITY = CONDUCTIVITY / (RHO_REFERENCE * CP)


def analytic_temperature(x: float, time_value: float) -> float:
    theta = x
    for mode in range(1, 401):
        theta += (
            2.0
            * ((-1.0) ** mode)
            * math.sin(mode * math.pi * x)
            * math.exp(-(mode * math.pi) ** 2 * THERMAL_DIFFUSIVITY * time_value)
            / (mode * math.pi)
        )
    return T_LEFT + (T_RIGHT - T_LEFT) * theta


time_value, time_dir = latest_time()
temperature = scalars(time_dir / "T")
density = scalars(time_dir / "rho")
nut = scalars(time_dir / "nut")
velocity = vectors(time_dir / "U")
expected = [analytic_temperature((i + 0.5) / N, time_value) for i in range(N)]
errors = [abs(a - b) for a, b in zip(temperature, expected)]
l2_error = math.sqrt(sum(error * error for error in errors) / N)
speed_max = max(math.sqrt(sum(component * component for component in value)) for value in velocity)
if not all(math.isfinite(value) and value > 0.0 for value in temperature + density):
    raise RuntimeError("non-finite or non-positive thermodynamic field")
if l2_error > 0.035 or max(errors) > 0.1 or speed_max > 0.05:
    raise RuntimeError("transient Fourier diffusion does not match the analytic solution")
if max(abs(value) for value in nut) > 1.0e-14:
    raise RuntimeError("laminar mode produced non-zero nut")
if max(temperature) < T_LEFT + 0.1:
    raise RuntimeError("wall heat did not diffuse into the gas")

report = {
    "time": time_dir.name,
    "conductivity": CONDUCTIVITY,
    "thermalDiffusivity": THERMAL_DIFFUSIVITY,
    "l2TemperatureError": l2_error,
    "maxTemperatureError": max(errors),
    "maxSpeed": speed_max,
    "maxAbsNut": max(abs(value) for value in nut),
}
(CASE / "check_fourier.json").write_text(json.dumps(report, indent=2) + "\n")
with (CASE / "validation_data.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(("x_m", "temperature_exact_K", "temperature_numerical_K"))
    for index, (target, value) in enumerate(zip(expected, temperature)):
        writer.writerow(((index + 0.5) / N, target, value))
print(json.dumps(report, indent=2))
