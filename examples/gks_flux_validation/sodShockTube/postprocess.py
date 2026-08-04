#!/usr/bin/env python3
"""Generate the Sod shock-tube validation plot from case-local CSV data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE = Path(__file__).resolve().parent
DATA = CASE / "validation_data.csv"
OUTPUT = CASE / "验证结果.png"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 22,
        "legend.fontsize": 15,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


def column(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


with DATA.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

x = column(rows, "x_m")
series = (
    ("rho_exact_kg_m3", "rho_numerical_kg_m3", r"Density $\rho$ ($\mathrm{kg\,m^{-3}}$)"),
    ("velocity_exact_m_s", "velocity_numerical_m_s", r"Velocity $U_x$ ($\mathrm{m\,s^{-1}}$)"),
    ("pressure_exact_Pa", "pressure_numerical_Pa", r"Pressure $p$ ($\mathrm{Pa}$)"),
)
figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
for axis, (exact_name, numerical_name, ylabel) in zip(axes, series):
    axis.plot(x, column(rows, exact_name), color="black", linewidth=2.6, label="Exact solution")
    axis.plot(
        x,
        column(rows, numerical_name),
        color="#C44E52",
        linewidth=1.6,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.9,
        markersize=4.8,
        markevery=8,
        label="Numerical result",
    )
    axis.set_xlabel(r"Axial position $x$ ($\mathrm{m}$)")
    axis.set_ylabel(ylabel)
    axis.tick_params(top=True, right=True, length=6, width=1.0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
axes[0].legend(frameon=False, loc="best")
figure.tight_layout(pad=0.7, w_pad=1.0)
figure.savefig(OUTPUT, dpi=600, facecolor="white")
plt.close(figure)
print(f"wrote={OUTPUT}")
