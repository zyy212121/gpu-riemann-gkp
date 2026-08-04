#!/usr/bin/env python3
"""Validate the wind-sand shock tube and generate a case-local plot."""

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


def main() -> None:
    parameters = json.loads((CASE / "reference/case_parameters.json").read_text(encoding="utf-8"))
    n_cells = int(parameters["n_cells"])
    final_time = float(parameters["final_time"])
    final_dir = CASE / f"{final_time:g}"
    x = (np.arange(n_cells) + 0.5) * parameters["length"] / n_cells
    published = np.genfromtxt(CASE / "reference/published/yang2022_collisionless_ugkwp.csv", delimiter=",", names=True)
    numerical = {
        "rho_g": read_scalar_field(final_dir / "rho", n_cells),
        "rho_p": read_scalar_field(final_dir / "epsilonS", n_cells) * 1000.0,
        "u_g": read_vector_field(final_dir / "U", n_cells)[:, 0],
        "u_p": read_vector_field(final_dir / "Us", n_cells)[:, 0],
        "p_g": read_scalar_field(final_dir / "p", n_cells),
        "T_p": read_scalar_field(final_dir / "Tp", n_cells),
    }
    gas_total_energy = read_scalar_field(final_dir / "rhoE", n_cells)
    particle_mechanical_energy = read_scalar_field(final_dir / "rhoEs", n_cells)
    reference = {name: np.interp(x, published["x"], published[name]) for name in numerical}
    normalized_mae = {}
    relative_l2 = {}
    correlation = {}
    extrema = {}
    for name in numerical:
        normalized_mae[name] = float(np.mean(np.abs(numerical[name] - reference[name])) / np.ptp(reference[name]))
        relative_l2[name] = float(np.linalg.norm(numerical[name] - reference[name]) / np.linalg.norm(reference[name]))
        correlation[name] = float(np.corrcoef(numerical[name], reference[name])[0, 1])
        extrema[name] = {"numerical_min": float(np.min(numerical[name])), "numerical_max": float(np.max(numerical[name])), "published_min": float(np.min(reference[name])), "published_max": float(np.max(reference[name]))}
    all_finite = all(np.all(np.isfinite(values)) for values in numerical.values())
    positive = bool(np.min(numerical["rho_g"]) > 0.0 and np.min(numerical["rho_p"]) > 0.0 and np.min(numerical["p_g"]) > 0.0 and np.min(numerical["T_p"]) > 0.0)
    cell_volume = parameters["length"] / n_cells
    particle_cp = parameters["particle_heat_capacity"]
    gas_mass_initial = 0.5 * parameters["length"] * (parameters["gas_left"]["rho"] + parameters["gas_right"]["rho"])
    particle_mass_initial = parameters["solid_apparent_density"] * parameters["length"]
    gas_energy_initial = 0.5 * parameters["length"] * (parameters["gas_left"]["p"] + parameters["gas_right"]["p"]) / (parameters["gamma"] - 1.0)
    particle_mechanical_initial = 0.5 * parameters["particle_fluctuation_dimensions"] * parameters["solid_pressure"] * parameters["length"]
    particle_material_initial = 0.5 * parameters["solid_apparent_density"] * parameters["length"] * particle_cp * (parameters["particle_temperature_left"] + parameters["particle_temperature_right"])
    initial_energy = gas_energy_initial + particle_mechanical_initial + particle_material_initial
    gas_mass_final = float(np.sum(numerical["rho_g"]) * cell_volume)
    particle_mass_final = float(np.sum(numerical["rho_p"]) * cell_volume)
    gas_energy_final = float(np.sum(gas_total_energy) * cell_volume)
    particle_mechanical_final = float(np.sum(particle_mechanical_energy) * cell_volume)
    particle_material_final = float(np.sum(numerical["rho_p"] * particle_cp * numerical["T_p"]) * cell_volume)
    final_energy = gas_energy_final + particle_mechanical_final + particle_material_final
    conservation = {
        "gas_mass_relative_change": (gas_mass_final - gas_mass_initial) / gas_mass_initial,
        "particle_mass_relative_change": (particle_mass_final - particle_mass_initial) / particle_mass_initial,
        "stored_energy_relative_change": (final_energy - initial_energy) / initial_energy,
    }
    tolerance = 0.15
    passed = bool(all_finite and positive and max(normalized_mae.values()) <= tolerance)
    result = {"case": "windSandShockTube", "source_doi": parameters["source_doi"], "final_time": final_time, "metrics": {"normalized_mean_absolute_error": normalized_mae, "relative_l2": relative_l2, "profile_correlation": correlation, "extrema": extrema, "all_values_finite": all_finite, "positive_density_pressure_temperature": positive, "conservation": conservation}, "tolerance": tolerance, "passed": passed}
    (CASE / "check_windsand.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    names = list(numerical)
    with (CASE / "validation_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_m"] + [f"{name}_published" for name in names] + [f"{name}_numerical" for name in names])
        for index, coordinate in enumerate(x):
            writer.writerow([coordinate] + [reference[name][index] for name in names] + [numerical[name][index] for name in names])

    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "mathtext.fontset": "stix", "axes.unicode_minus": False})
    labels = {"rho_g": r"Apparent gas density $\rho_g$", "rho_p": r"Apparent particle density $\rho_p$", "u_g": r"Gas velocity $U_g$", "u_p": r"Particle velocity $U_p$", "p_g": r"Gas pressure $p_g$", "T_p": r"Particle temperature $T_p$"}
    figure, axes = plt.subplots(3, 2, figsize=(7.2, 7.2), sharex=True)
    for axis, name in zip(axes.flat, numerical):
        axis.plot(x, reference[name], color="black", linewidth=1.5, label="Published result")
        axis.plot(x, numerical[name], color="#C44E52", linewidth=1.1, marker="o", markersize=3.0, markerfacecolor="white", markeredgewidth=0.7, markevery=3, label="Numerical result")
        axis.set_ylabel(labels[name])
        axis.set_xlim(0.0, parameters["length"])
        axis.tick_params(top=True, right=True, length=5, width=0.9)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"Axial position $x$ ($\mathrm{m}$)")
    axes[0, 0].legend(frameon=False, loc="best")
    figure.tight_layout(pad=0.7, h_pad=0.35)
    figure.savefig(CASE / "验证结果.png", dpi=600, facecolor="white")
    plt.close(figure)
    print(json.dumps({"normalized_mean_absolute_error": normalized_mae, "conservation": conservation, "passed": passed}, indent=2))
    print(f"wrote={CASE / '验证结果.png'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
