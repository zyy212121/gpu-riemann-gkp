#!/usr/bin/env python3
"""Validate homogeneous gas-particle relaxation and generate a local plot."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE = Path(__file__).resolve().parent
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def numeric_time_directories() -> list[tuple[float, Path]]:
    result = []
    for child in CASE.iterdir():
        if child.is_dir():
            try:
                result.append((float(child.name), child))
            except ValueError:
                pass
    return sorted(result)


def read_vector_field(path: Path, expected_size: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    uniform = re.search(
        rf"internalField\s+uniform\s*\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)\s*;",
        text,
    )
    if uniform:
        vector = np.array([float(uniform.group(i)) for i in range(1, 4)])
        return np.repeat(vector[None, :], expected_size, axis=0)
    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"Cannot read vector field {path}")
    rows = re.findall(rf"\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)", match.group(2))
    values = np.asarray(rows, dtype=float)
    if values.shape != (expected_size, 3) or int(match.group(1)) != expected_size:
        raise ValueError(f"Unexpected cell count in {path}")
    return values


def exact_velocities(time: np.ndarray, parameters: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    rho_g = parameters["rho_g"]
    rho_p = parameters["rho_p"]
    rho_solid = parameters["rho_solid"]
    diameter = parameters["diameter"]
    viscosity = parameters["viscosity"]
    gas_initial = parameters["gas_velocity_initial"]
    particle_initial = parameters["particle_velocity_initial"]
    exponent = 0.687
    relative_initial = gas_initial - particle_initial
    stokes_rate = 18.0 * viscosity / (rho_solid * diameter**2)
    nonlinear_factor = 0.15 * (rho_g * diameter / viscosity) ** exponent
    coupled_rate = (1.0 + rho_p / rho_g) * stokes_rate
    relative = relative_initial * np.exp(-coupled_rate * time) / (
        1.0
        + nonlinear_factor
        * abs(relative_initial) ** exponent
        * (1.0 - np.exp(-exponent * coupled_rate * time))
    ) ** (1.0 / exponent)
    mixture_velocity = (rho_g * gas_initial + rho_p * particle_initial) / (rho_g + rho_p)
    gas = mixture_velocity + rho_p / (rho_g + rho_p) * relative
    particle = mixture_velocity - rho_g / (rho_g + rho_p) * relative
    return gas, particle


def main() -> None:
    parameters = json.loads((CASE / "reference/case_parameters.json").read_text(encoding="utf-8"))
    n_cells = int(parameters["n_cells"])
    cell_begin, cell_end = parameters["analysis_cells"]
    times: list[float] = []
    gas_mean: list[float] = []
    particle_mean: list[float] = []
    gas_std: list[float] = []
    particle_std: list[float] = []
    for time, directory in numeric_time_directories():
        if not (directory / "U").is_file() or not (directory / "Us").is_file():
            continue
        gas = read_vector_field(directory / "U", n_cells)[cell_begin:cell_end, 0]
        particle = read_vector_field(directory / "Us", n_cells)[cell_begin:cell_end, 0]
        times.append(time)
        gas_mean.append(float(np.mean(gas)))
        particle_mean.append(float(np.mean(particle)))
        gas_std.append(float(np.std(gas)))
        particle_std.append(float(np.std(particle)))
    if not times:
        raise RuntimeError("No time directory is available for post-processing")
    time_array = np.asarray(times)
    gas_array = np.asarray(gas_mean)
    particle_array = np.asarray(particle_mean)
    exact_gas, exact_particle = exact_velocities(time_array, parameters)
    gas_error = np.abs(gas_array - exact_gas)
    particle_error = np.abs(particle_array - exact_particle)
    rho_g = parameters["rho_g"]
    rho_p = parameters["rho_p"]
    initial_momentum = rho_g * parameters["gas_velocity_initial"] + rho_p * parameters["particle_velocity_initial"]
    momentum_error = np.abs(rho_g * gas_array + rho_p * particle_array - initial_momentum)
    tolerances = {"maximum_velocity_absolute_error": 2.0e-3, "maximum_interior_standard_deviation": 2.0e-3, "maximum_momentum_absolute_error": 2.0e-3}
    metrics = {
        "samples": len(times),
        "final_time": float(time_array[-1]),
        "gas_velocity_max_abs_error": float(np.max(gas_error)),
        "particle_velocity_max_abs_error": float(np.max(particle_error)),
        "gas_interior_std_max": float(np.max(gas_std)),
        "particle_interior_std_max": float(np.max(particle_std)),
        "mixture_momentum_max_abs_error": float(np.max(momentum_error)),
    }
    passed = max(metrics["gas_velocity_max_abs_error"], metrics["particle_velocity_max_abs_error"]) <= tolerances["maximum_velocity_absolute_error"] and max(metrics["gas_interior_std_max"], metrics["particle_interior_std_max"]) <= tolerances["maximum_interior_standard_deviation"] and metrics["mixture_momentum_max_abs_error"] <= tolerances["maximum_momentum_absolute_error"]
    result = {"case": "dustyBox", "reference": "nonlinear Schiller-Naumann homogeneous relaxation", "metrics": metrics, "tolerances": tolerances, "passed": passed}
    (CASE / "check_dustybox.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (CASE / "validation_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "gas_velocity_exact_m_s", "gas_velocity_numerical_m_s", "particle_velocity_exact_m_s", "particle_velocity_numerical_m_s", "gas_velocity_abs_error_m_s", "particle_velocity_abs_error_m_s", "mixture_momentum_abs_error"))
        writer.writerows(zip(time_array, exact_gas, gas_array, exact_particle, particle_array, gas_error, particle_error, momentum_error))

    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "mathtext.fontset": "stix", "axes.unicode_minus": False})
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    axes[0].plot(time_array, exact_gas, color="black", linewidth=2.5, label="Analytical gas")
    axes[0].plot(time_array, exact_particle, color="black", linewidth=2.5, linestyle="--", label="Analytical particle")
    axes[0].plot(time_array, gas_array, color="#C44E52", linewidth=1.6, marker="o", markersize=4.8, markerfacecolor="white", markeredgewidth=0.9, markevery=2, label="Numerical gas")
    axes[0].plot(time_array, particle_array, color="#4C72B0", linewidth=1.6, marker="s", markersize=4.8, markerfacecolor="white", markeredgewidth=0.9, markevery=2, label="Numerical particle")
    axes[0].set_xlabel(r"Time $t$ ($\mathrm{s}$)")
    axes[0].set_ylabel(r"Axial velocity $U_x$ ($\mathrm{m\,s^{-1}}$)")
    axes[0].legend(frameon=False, loc="best")
    axes[1].semilogy(time_array, np.maximum(gas_error, 1.0e-16), color="#C44E52", linewidth=1.6, label="Gas velocity error")
    axes[1].semilogy(time_array, np.maximum(particle_error, 1.0e-16), color="#4C72B0", linewidth=1.6, linestyle="--", label="Particle velocity error")
    axes[1].semilogy(time_array, np.maximum(momentum_error, 1.0e-16), color="#55A868", linewidth=1.6, linestyle="-.", label="Mixture momentum error")
    axes[1].set_xlabel(r"Time $t$ ($\mathrm{s}$)")
    axes[1].set_ylabel("Absolute error")
    axes[1].legend(frameon=False, loc="best")
    for axis in axes:
        axis.tick_params(top=True, right=True, length=6, width=1.0)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
    figure.tight_layout(pad=0.7, w_pad=1.0)
    figure.savefig(CASE / "验证结果.png", dpi=600, facecolor="white")
    plt.close(figure)
    print(json.dumps({"metrics": metrics, "passed": passed}, indent=2))
    print(f"wrote={CASE / '验证结果.png'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
