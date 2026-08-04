#!/usr/bin/env python3
"""Measure observed order for a smooth standing acoustic mode after one step."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GAMMA = 1.4
R_GAS = 287.0
P0 = 100000.0
T0 = 300.0
RHO0 = P0 / (R_GAS * T0)
SOUND = math.sqrt(GAMMA * P0 / RHO0)
AMPLITUDE = 1.0
PHASE = math.pi / 4.0
WAVENUMBER = math.pi
OMEGA = SOUND * WAVENUMBER
DT = 1.0e-6


def latest(case: Path) -> Path:
    times = []
    for path in case.iterdir():
        try:
            time = float(path.name)
        except ValueError:
            continue
        if path.is_dir() and time > 0.0:
            times.append((time, path))
    if not times:
        raise RuntimeError(f"no result time in {case}")
    time, path = max(times)
    if not math.isclose(time, DT, rel_tol=0.0, abs_tol=1.0e-14):
        raise RuntimeError(f"unexpected acoustic result time {time}")
    return path


def scalars(path: Path, expected: int) -> list[float]:
    text = path.read_text()
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if not match:
        raise RuntimeError(f"cannot parse {path}")
    values = [float(value) for value in match.group(2).split()]
    if int(match.group(1)) != expected or len(values) != expected:
        raise RuntimeError(f"wrong field size in {path}")
    return values


def velocity_x(path: Path, expected: int) -> list[float]:
    text = path.read_text()
    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if not match:
        raise RuntimeError(f"cannot parse {path}")
    values = [
        float(item.split()[0])
        for item in re.findall(r"\(([^()]*)\)", match.group(2))
    ]
    if int(match.group(1)) != expected or len(values) != expected:
        raise RuntimeError(f"wrong vector size in {path}")
    return values


profile_rows: list[dict[str, float | int | str]] = []


def errors(cells: int) -> dict[str, float]:
    directory = latest(ROOT / f"N{cells}")
    pressure = scalars(directory / "p", cells)
    density = scalars(directory / "rho", cells)
    velocity = velocity_x(directory / "U", cells)
    phase = PHASE + OMEGA * DT
    dx = 1.0 / cells
    factor = math.sin(0.5 * WAVENUMBER * dx) / (0.5 * WAVENUMBER * dx)
    p_error = 0.0
    rho_error = 0.0
    u_error = 0.0
    for cell in range(cells):
        x = (cell + 0.5) * dx
        p_prime = AMPLITUDE * factor * math.cos(WAVENUMBER * x) * math.cos(phase)
        p_exact = P0 + p_prime
        rho_exact = RHO0 + p_prime / (SOUND * SOUND)
        u_exact = (
            AMPLITUDE / (RHO0 * SOUND) * factor
            * math.sin(WAVENUMBER * x) * math.sin(phase)
        )
        p_error += abs(pressure[cell] - p_exact)
        rho_error += abs(density[cell] - rho_exact)
        u_error += abs(velocity[cell] - u_exact)
        profile_rows.append(
            {
                "record": "profile",
                "cells": cells,
                "x_m": x,
                "pressure_exact_Pa": p_exact - P0,
                "pressure_numerical_Pa": pressure[cell] - P0,
                "pressure_relative_l1": "",
                "density_relative_l1": "",
                "velocity_relative_l1": "",
            }
        )
    return {
        "p": p_error / cells / AMPLITUDE,
        "rho": rho_error / cells / (AMPLITUDE / (SOUND * SOUND)),
        "U": u_error / cells / (AMPLITUDE / (RHO0 * SOUND)),
    }


rows = {str(cells): errors(cells) for cells in (50, 100, 200)}
orders: dict[str, list[float]] = {}
for field in ("p", "rho", "U"):
    orders[field] = [
        math.log(rows["50"][field] / rows["100"][field], 2.0),
        math.log(rows["100"][field] / rows["200"][field], 2.0),
    ]
if not all(math.isfinite(value) for row in rows.values() for value in row.values()):
    raise RuntimeError("non-finite acoustic error")
if min(orders["p"] + orders["rho"] + orders["U"]) < 1.7:
    raise RuntimeError(f"second-order acoustic convergence not observed: {orders}")
report = {"oneStepDeltaT": DT, "errors": rows, "observedOrders": orders}
(ROOT / "check_acoustic.json").write_text(json.dumps(report, indent=2) + "\n")
with (ROOT / "validation_data.csv").open("w", newline="", encoding="utf-8") as stream:
    fieldnames = (
        "record",
        "cells",
        "x_m",
        "pressure_exact_Pa",
        "pressure_numerical_Pa",
        "pressure_relative_l1",
        "density_relative_l1",
        "velocity_relative_l1",
    )
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(profile_rows)
    for cells in (50, 100, 200):
        metrics = rows[str(cells)]
        writer.writerow(
            {
                "record": "error",
                "cells": cells,
                "x_m": "",
                "pressure_exact_Pa": "",
                "pressure_numerical_Pa": "",
                "pressure_relative_l1": metrics["p"],
                "density_relative_l1": metrics["rho"],
                "velocity_relative_l1": metrics["U"],
            }
        )
print(json.dumps(report, indent=2))
