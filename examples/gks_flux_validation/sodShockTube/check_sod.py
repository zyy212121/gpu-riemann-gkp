#!/usr/bin/env python3
"""Check conservation and compare the Sod result with the exact Riemann solution."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


CASE = Path(__file__).resolve().parent
N_CELLS = 200
GAMMA = 1.4
LEFT = (1.0, 0.0, 100000.0)
RIGHT = (0.125, 0.0, 10000.0)
CELL_VOLUME = (1.0 / N_CELLS) * 0.01 * 0.01
INITIAL_MASS = 0.5 * (LEFT[0] + RIGHT[0]) * 0.01 * 0.01
INITIAL_ENERGY = 0.5 * (LEFT[2] + RIGHT[2]) / (GAMMA - 1.0) * 0.01 * 0.01


def latest_time() -> tuple[float, Path]:
    candidates: list[tuple[float, Path]] = []
    for path in CASE.iterdir():
        if not path.is_dir() or path.name == "0":
            continue
        try:
            candidates.append((float(path.name), path))
        except ValueError:
            pass
    if not candidates:
        raise RuntimeError("no result time directory was written")
    return max(candidates)


def scalar_values(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if match:
        values = [float(token) for token in match.group(2).split()]
        if len(values) != int(match.group(1)):
            raise RuntimeError(f"invalid list length in {path}")
        return values
    match = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if not match:
        raise RuntimeError(f"cannot parse internalField in {path}")
    return [float(match.group(1))] * N_CELLS


def vector_values(path: Path) -> list[tuple[float, float, float]]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if match:
        values = [
            tuple(float(v) for v in item.split())
            for item in re.findall(r"\(([^()]*)\)", match.group(2))
        ]
        if len(values) != int(match.group(1)):
            raise RuntimeError(f"invalid vector list length in {path}")
        return values
    match = re.search(r"internalField\s+uniform\s+\(([^()]*)\)\s*;", text)
    if not match:
        raise RuntimeError(f"cannot parse vector internalField in {path}")
    value = tuple(float(v) for v in match.group(1).split())
    return [value] * N_CELLS


def pressure_function(p: float, state: tuple[float, float, float]) -> tuple[float, float]:
    rho, _, pressure = state
    sound = math.sqrt(GAMMA * pressure / rho)
    if p > pressure:
        a = 2.0 / ((GAMMA + 1.0) * rho)
        b = pressure * (GAMMA - 1.0) / (GAMMA + 1.0)
        root = math.sqrt(a / (p + b))
        value = (p - pressure) * root
        derivative = root * (1.0 - 0.5 * (p - pressure) / (p + b))
        return value, derivative
    exponent = (GAMMA - 1.0) / (2.0 * GAMMA)
    ratio = p / pressure
    value = 2.0 * sound / (GAMMA - 1.0) * (ratio**exponent - 1.0)
    derivative = ratio ** (-(GAMMA + 1.0) / (2.0 * GAMMA)) / (rho * sound)
    return value, derivative


def star_state() -> tuple[float, float]:
    rho_l, u_l, p_l = LEFT
    rho_r, u_r, p_r = RIGHT
    a_l = math.sqrt(GAMMA * p_l / rho_l)
    a_r = math.sqrt(GAMMA * p_r / rho_r)
    guess = max(
        1.0e-12,
        0.5 * (p_l + p_r)
        - 0.125 * (u_r - u_l) * (rho_l + rho_r) * (a_l + a_r),
    )
    pressure = guess
    for _ in range(80):
        f_l, d_l = pressure_function(pressure, LEFT)
        f_r, d_r = pressure_function(pressure, RIGHT)
        updated = pressure - (f_l + f_r + u_r - u_l) / (d_l + d_r)
        updated = max(updated, 1.0e-12)
        if abs(updated - pressure) <= 1.0e-12 * max(updated, pressure):
            pressure = updated
            break
        pressure = updated
    f_l, _ = pressure_function(pressure, LEFT)
    f_r, _ = pressure_function(pressure, RIGHT)
    velocity = 0.5 * (u_l + u_r + f_r - f_l)
    return pressure, velocity


P_STAR, U_STAR = star_state()


def exact_state(x: float, time_value: float) -> tuple[float, float, float]:
    xi = (x - 0.5) / time_value
    rho_l, u_l, p_l = LEFT
    rho_r, u_r, p_r = RIGHT
    a_l = math.sqrt(GAMMA * p_l / rho_l)
    a_r = math.sqrt(GAMMA * p_r / rho_r)
    if xi <= U_STAR:
        if P_STAR > p_l:
            shock = u_l - a_l * math.sqrt(
                (GAMMA + 1.0) * P_STAR / (2.0 * GAMMA * p_l)
                + (GAMMA - 1.0) / (2.0 * GAMMA)
            )
            if xi <= shock:
                return LEFT
            ratio = P_STAR / p_l
            rho = rho_l * (
                (ratio + (GAMMA - 1.0) / (GAMMA + 1.0))
                / ((GAMMA - 1.0) * ratio / (GAMMA + 1.0) + 1.0)
            )
            return rho, U_STAR, P_STAR
        a_star = a_l * (P_STAR / p_l) ** ((GAMMA - 1.0) / (2.0 * GAMMA))
        head = u_l - a_l
        tail = U_STAR - a_star
        if xi <= head:
            return LEFT
        if xi >= tail:
            rho = rho_l * (P_STAR / p_l) ** (1.0 / GAMMA)
            return rho, U_STAR, P_STAR
        velocity = 2.0 * (a_l + 0.5 * (GAMMA - 1.0) * u_l + xi) / (GAMMA + 1.0)
        sound = 2.0 * (a_l + 0.5 * (GAMMA - 1.0) * (u_l - xi)) / (GAMMA + 1.0)
        rho = rho_l * (sound / a_l) ** (2.0 / (GAMMA - 1.0))
        pressure = p_l * (sound / a_l) ** (2.0 * GAMMA / (GAMMA - 1.0))
        return rho, velocity, pressure

    if P_STAR > p_r:
        shock = u_r + a_r * math.sqrt(
            (GAMMA + 1.0) * P_STAR / (2.0 * GAMMA * p_r)
            + (GAMMA - 1.0) / (2.0 * GAMMA)
        )
        if xi >= shock:
            return RIGHT
        ratio = P_STAR / p_r
        rho = rho_r * (
            (ratio + (GAMMA - 1.0) / (GAMMA + 1.0))
            / ((GAMMA - 1.0) * ratio / (GAMMA + 1.0) + 1.0)
        )
        return rho, U_STAR, P_STAR

    a_star = a_r * (P_STAR / p_r) ** ((GAMMA - 1.0) / (2.0 * GAMMA))
    head = u_r + a_r
    tail = U_STAR + a_star
    if xi >= head:
        return RIGHT
    if xi <= tail:
        rho = rho_r * (P_STAR / p_r) ** (1.0 / GAMMA)
        return rho, U_STAR, P_STAR
    velocity = 2.0 * (-a_r + 0.5 * (GAMMA - 1.0) * u_r + xi) / (GAMMA + 1.0)
    sound = 2.0 * (a_r - 0.5 * (GAMMA - 1.0) * (u_r - xi)) / (GAMMA + 1.0)
    rho = rho_r * (sound / a_r) ** (2.0 / (GAMMA - 1.0))
    pressure = p_r * (sound / a_r) ** (2.0 * GAMMA / (GAMMA - 1.0))
    return rho, velocity, pressure


time_value, time_dir = latest_time()
rho = scalar_values(time_dir / "rho")
p = scalar_values(time_dir / "p")
rho_e = scalar_values(time_dir / "rhoE")
velocity = vector_values(time_dir / "U")
if not (len(rho) == len(p) == len(rho_e) == len(velocity) == N_CELLS):
    raise RuntimeError("unexpected cell count")
if not all(math.isfinite(value) and value > 0.0 for value in rho + p + rho_e):
    raise RuntimeError("non-finite or non-positive gas state")
if not all(math.isfinite(v) for item in velocity for v in item):
    raise RuntimeError("non-finite velocity")

mass = sum(rho) * CELL_VOLUME
energy = sum(rho_e) * CELL_VOLUME
mass_error = abs(mass - INITIAL_MASS) / INITIAL_MASS
energy_error = abs(energy - INITIAL_ENERGY) / INITIAL_ENERGY
if mass_error > 2.0e-10 or energy_error > 2.0e-10:
    raise RuntimeError(
        f"closed-tube conservation failed: mass={mass_error:.3e}, "
        f"energy={energy_error:.3e}"
    )

exact = [exact_state((i + 0.5) / N_CELLS, time_value) for i in range(N_CELLS)]
rho_l1 = sum(abs(value - target[0]) for value, target in zip(rho, exact)) / N_CELLS
p_l1 = sum(abs(value - target[2]) for value, target in zip(p, exact)) / N_CELLS
u_l1 = sum(abs(value[0] - target[1]) for value, target in zip(velocity, exact)) / N_CELLS
rho_rel = rho_l1 / max(value[0] for value in exact)
p_rel = p_l1 / max(value[2] for value in exact)
u_rel = u_l1 / max(abs(value[1]) for value in exact)
cross_velocity = max(max(abs(item[1]), abs(item[2])) for item in velocity)
if rho_rel > 0.08 or p_rel > 0.08 or u_rel > 0.12 or cross_velocity > 1.0e-8:
    raise RuntimeError("Sod density/pressure/velocity do not match the exact wave pattern")

report = {
    "time": time_dir.name,
    "pStar": P_STAR,
    "uStar": U_STAR,
    "massRelativeError": mass_error,
    "energyRelativeError": energy_error,
    "rhoRelativeL1": rho_rel,
    "pRelativeL1": p_rel,
    "uRelativeL1": u_rel,
    "maxCrossVelocity": cross_velocity,
}
(CASE / "check_sod.json").write_text(json.dumps(report, indent=2) + "\n")
with (CASE / "validation_data.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(
        (
            "x_m",
            "rho_exact_kg_m3",
            "rho_numerical_kg_m3",
            "velocity_exact_m_s",
            "velocity_numerical_m_s",
            "pressure_exact_Pa",
            "pressure_numerical_Pa",
        )
    )
    for index, (numerical_rho, numerical_u, numerical_p, target) in enumerate(
        zip(rho, velocity, p, exact)
    ):
        writer.writerow(
            (
                (index + 0.5) / N_CELLS,
                target[0],
                numerical_rho,
                target[1],
                numerical_u[0],
                target[2],
                numerical_p,
            )
        )
print(json.dumps(report, indent=2))
