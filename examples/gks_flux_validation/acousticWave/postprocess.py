#!/usr/bin/env python3
"""Generate the acoustic-wave convergence plot from case-local CSV data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "serif", "font.size": 18, "axes.labelsize": 22, "legend.fontsize": 15, "xtick.labelsize": 18, "ytick.labelsize": 18, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "mathtext.fontset": "stix", "axes.unicode_minus": False})

with (CASE / "validation_data.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

profiles = [row for row in rows if row["record"] == "profile"]
errors = [row for row in rows if row["record"] == "error"]
figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

exact_rows = [row for row in profiles if int(row["cells"]) == 200]
axes[0].plot(
    [float(row["x_m"]) for row in exact_rows],
    [float(row["pressure_exact_Pa"]) for row in exact_rows],
    color="black",
    linewidth=2.6,
    label="Analytical solution",
)
colors = {50: "#4C72B0", 100: "#55A868", 200: "#C44E52"}
markers = {50: "s", 100: "^", 200: "o"}
for cells in (50, 100, 200):
    selected = [row for row in profiles if int(row["cells"]) == cells]
    axes[0].plot(
        [float(row["x_m"]) for row in selected],
        [float(row["pressure_numerical_Pa"]) for row in selected],
        color=colors[cells],
        linewidth=1.5,
        marker=markers[cells],
        markerfacecolor="white",
        markeredgewidth=0.8,
        markersize=4.8,
        markevery=max(1, cells // 25),
        label=f"{cells} cells",
    )
axes[0].set_xlabel(r"Position $x$ ($\mathrm{m}$)")
axes[0].set_ylabel(r"Pressure perturbation $p'$ ($\mathrm{Pa}$)")
axes[0].legend(frameon=False, loc="best")

cells_values = [int(row["cells"]) for row in errors]
for field, label, color, marker in (
    ("pressure_relative_l1", "Pressure", "#C44E52", "o"),
    ("density_relative_l1", "Density", "#4C72B0", "s"),
    ("velocity_relative_l1", "Velocity", "#55A868", "^"),
):
    axes[1].loglog(
        cells_values,
        [float(row[field]) for row in errors],
        color=color,
        linewidth=1.6,
        marker=marker,
        markerfacecolor="white",
        markeredgewidth=0.9,
        markersize=5.0,
        label=label,
    )
reference = float(errors[0]["pressure_relative_l1"])
axes[1].loglog(
    cells_values,
    [reference * (cells_values[0] / value) ** 2 for value in cells_values],
    color="black",
    linestyle="--",
    linewidth=2.0,
    label="Second-order reference",
)
axes[1].set_xlabel(r"Number of cells $N$")
axes[1].set_ylabel(r"Relative $L_1$ error")
axes[1].legend(frameon=False, loc="best")

for axis in axes:
    axis.tick_params(top=True, right=True, length=6, width=1.0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
figure.tight_layout(pad=0.7, w_pad=1.0)
figure.savefig(CASE / "验证结果.png", dpi=600, facecolor="white")
plt.close(figure)
print(f"wrote={CASE / '验证结果.png'}")
