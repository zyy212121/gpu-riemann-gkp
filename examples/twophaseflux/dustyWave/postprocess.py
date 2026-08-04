#!/usr/bin/env python3
"""Validate a linear gas-particle acoustic wave and generate a local plot."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.linalg import expm

CASE = Path(__file__).resolve().parent
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def read_scalar_field(path: Path, expected_size: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    uniform = re.search(rf"internalField\s+uniform\s+({NUMBER})\s*;", text)
    if uniform:
        return np.full(expected_size, float(uniform.group(1)))
    match = re.search(r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;", text, re.DOTALL)
    if not match:
        raise ValueError(f"Cannot read scalar field {path}")
    values = np.fromstring(match.group(2), sep=" ")
    if values.size != expected_size or int(match.group(1)) != expected_size:
        raise ValueError(f"Unexpected cell count in {path}")
    return values


def read_vector_field(path: Path, expected_size: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    uniform = re.search(rf"internalField\s+uniform\s*\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)\s*;", text)
    if uniform:
        vector = np.array([float(uniform.group(i)) for i in range(1, 4)])
        return np.repeat(vector[None, :], expected_size, axis=0)
    match = re.search(r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;", text, re.DOTALL)
    if not match:
        raise ValueError(f"Cannot read vector field {path}")
    rows = re.findall(rf"\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)", match.group(2))
    values = np.asarray(rows, dtype=float)
    if values.shape != (expected_size, 3) or int(match.group(1)) != expected_size:
        raise ValueError(f"Unexpected cell count in {path}")
    return values


def linear_reference(x: np.ndarray, time: float, parameters: dict[str, float]) -> dict[str, np.ndarray]:
    amplitude = parameters["amplitude"]
    wave_number = parameters["wave_number"]
    rho_g0 = parameters["rho_g0"]
    rho_p0 = parameters["rho_p0"]
    p0 = parameters["p0"]
    gamma = parameters["gamma"]
    response_time = parameters["response_time"]
    sound_speed_squared = gamma * p0 / rho_g0
    ratio = rho_p0 / rho_g0
    system = np.array(
        [
            [0.0, 0.0, -1j * wave_number * rho_g0, 0.0],
            [0.0, 0.0, 0.0, -1j * wave_number * rho_p0],
            [-1j * wave_number * sound_speed_squared / rho_g0, 0.0, -ratio / response_time, ratio / response_time],
            [0.0, 0.0, 1.0 / response_time, -1.0 / response_time],
        ],
        dtype=complex,
    )
    initial_hat = -1j * amplitude * np.array([rho_g0, rho_p0, 1.0, 1.0], dtype=complex)
    state_hat = expm(system * time) @ initial_hat
    perturbations = np.real(state_hat[:, None] * np.exp(1j * wave_number * x)[None, :])
    return {"rho": rho_g0 + perturbations[0], "rhoP": rho_p0 + perturbations[1], "U": perturbations[2], "Us": perturbations[3], "p": p0 + sound_speed_squared * perturbations[0]}


def relative_l2(numerical: np.ndarray, exact: np.ndarray, base: float) -> float:
    return float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact - base))


def harmonic_error(x: np.ndarray, numerical: np.ndarray, exact: np.ndarray, base: float, wave_number: float) -> tuple[float, float]:
    kernel = np.exp(-1j * wave_number * x)
    numerical_hat = 2.0 * np.mean((numerical - base) * kernel)
    exact_hat = 2.0 * np.mean((exact - base) * kernel)
    return float(abs(abs(numerical_hat) - abs(exact_hat)) / abs(exact_hat)), float(abs(np.angle(numerical_hat / exact_hat)))


def main() -> None:
    parameters = json.loads((CASE / "reference/case_parameters.json").read_text(encoding="utf-8"))
    n_cells = int(parameters["n_cells"])
    length = parameters["length"]
    final_time = float(parameters.get("final_time", 0.2))
    final_dir = CASE / f"{final_time:g}"
    x = (np.arange(n_cells) + 0.5) * length / n_cells
    window_begin, window_end = parameters["analysis_window"]
    mask = (x >= window_begin) & (x < window_end)
    x_window = x[mask]
    numerical = {
        "rho": read_scalar_field(final_dir / "rho", n_cells)[mask],
        "rhoP": read_scalar_field(final_dir / "epsilonS", n_cells)[mask] * parameters["rho_solid"],
        "U": read_vector_field(final_dir / "U", n_cells)[mask, 0],
        "Us": read_vector_field(final_dir / "Us", n_cells)[mask, 0],
        "p": read_scalar_field(final_dir / "p", n_cells)[mask],
    }
    exact_all = linear_reference(x, final_time, parameters)
    exact = {name: values[mask] for name, values in exact_all.items()}
    bases = {"rho": parameters["rho_g0"], "rhoP": parameters["rho_p0"], "U": 0.0, "Us": 0.0, "p": parameters["p0"]}
    l2_errors = {name: relative_l2(numerical[name], exact[name], bases[name]) for name in numerical}
    harmonic = {}
    for name in numerical:
        amplitude_error, phase_error = harmonic_error(x_window, numerical[name], exact[name], bases[name], parameters["wave_number"])
        harmonic[name] = {"relative_amplitude_error": amplitude_error, "phase_error_radian": phase_error}
    mean_drift = {name: float(abs(np.mean(numerical[name]) - bases[name])) for name in numerical}
    minima = {name: float(np.min(values)) for name, values in numerical.items()}
    finite = all(np.all(np.isfinite(values)) for values in numerical.values())
    tolerances = {"maximum_relative_perturbation_l2": 0.15, "maximum_relative_harmonic_amplitude_error": 0.12, "maximum_harmonic_phase_error_radian": 0.10, "maximum_mean_drift": 5.0e-3}
    passed = finite and max(l2_errors.values()) <= tolerances["maximum_relative_perturbation_l2"] and max(item["relative_amplitude_error"] for item in harmonic.values()) <= tolerances["maximum_relative_harmonic_amplitude_error"] and max(item["phase_error_radian"] for item in harmonic.values()) <= tolerances["maximum_harmonic_phase_error_radian"] and max(mean_drift.values()) <= tolerances["maximum_mean_drift"] and minima["rho"] > 0.0 and minima["rhoP"] > 0.0 and minima["p"] > 0.0
    result = {"case": "dustyWave", "final_time": final_time, "metrics": {"relative_perturbation_l2": l2_errors, "harmonic": harmonic, "mean_drift": mean_drift, "minimum_values": minima, "all_values_finite": finite}, "tolerances": tolerances, "passed": passed}
    (CASE / "check_dustywave.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (CASE / "validation_data.csv").open("w", newline="", encoding="utf-8") as stream:
        names = list(numerical)
        writer = csv.writer(stream)
        writer.writerow(["x_m"] + [f"{name}_exact" for name in names] + [f"{name}_numerical" for name in names])
        for index, coordinate in enumerate(x_window):
            writer.writerow([coordinate] + [exact[name][index] for name in names] + [numerical[name][index] for name in names])

    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "mathtext.fontset": "stix", "axes.unicode_minus": False})
    figure = plt.figure(figsize=(7.2, 7.0))
    grid = GridSpec(3, 2, figure=figure)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1]), figure.add_subplot(grid[2, :])]
    labels = {"rho": r"Gas density perturbation $\rho_g-\rho_{g,0}$", "rhoP": r"Particle density perturbation $\rho_p-\rho_{p,0}$", "U": r"Gas velocity $U_g$", "Us": r"Particle velocity $U_p$", "p": r"Pressure perturbation $p_g-p_{g,0}$"}
    for axis, name in zip(axes, numerical):
        axis.plot(x_window, exact[name] - bases[name], color="black", linewidth=1.5, label="Linear analytical solution")
        axis.plot(x_window, numerical[name] - bases[name], color="#C44E52", linewidth=1.1, marker="o", markersize=3.0, markerfacecolor="white", markeredgewidth=0.7, markevery=4, label="Numerical result")
        axis.set_ylabel(labels[name])
        axis.set_xlim(window_begin, window_end)
        axis.tick_params(top=True, right=True, length=5, width=0.9)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
    axes[0].legend(frameon=False, loc="best")
    for axis in (axes[2], axes[3], axes[4]):
        axis.set_xlabel(r"Axial position $x$ ($\mathrm{m}$)")
    figure.tight_layout(pad=0.7, h_pad=0.35)
    figure.savefig(CASE / "验证结果.png", dpi=600, facecolor="white")
    plt.close(figure)
    print(json.dumps({"relative_perturbation_l2": l2_errors, "passed": passed}, indent=2))
    print(f"wrote={CASE / '验证结果.png'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
