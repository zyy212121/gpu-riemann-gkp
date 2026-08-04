#!/usr/bin/env python3
"""Generate the Pr=0.8 Fourier-conduction plot from case-local CSV data."""

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
x = [float(row["x_m"]) for row in rows]
exact = [float(row["temperature_exact_K"]) for row in rows]
numerical = [float(row["temperature_numerical_K"]) for row in rows]
figure, axis = plt.subplots(figsize=(10.0, 8.0))
axis.plot(x, exact, color="black", linewidth=2.6, label="Analytical solution")
axis.plot(x, numerical, color="#C44E52", linewidth=1.6, marker="o", markerfacecolor="white", markeredgewidth=0.9, markersize=5.0, markevery=4, label="Numerical result")
axis.set_xlabel(r"Position $x$ ($\mathrm{m}$)")
axis.set_ylabel(r"Temperature $T$ ($\mathrm{K}$)")
axis.tick_params(top=True, right=True, length=6, width=1.0)
axis.grid(False)
for spine in axis.spines.values():
    spine.set_linewidth(1.0)
axis.legend(frameon=False, loc="best")
figure.tight_layout()
figure.savefig(CASE / "验证结果.png", dpi=600, facecolor="white")
plt.close(figure)
print(f"wrote={CASE / '验证结果.png'}")
