#!/usr/bin/env python3
"""Generate the affine-flow eddy-viscosity validation plot."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "serif", "font.size": 18, "axes.labelsize": 22, "xtick.labelsize": 18, "ytick.labelsize": 18, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "mathtext.fontset": "stix", "axes.unicode_minus": False})
with (CASE / "validation_data.csv").open(newline="", encoding="utf-8") as stream:
    row = next(csv.DictReader(stream))
values = [float(row["nut_exact_m2_s"]), float(row["nut_numerical_m2_s"])]
figure, axis = plt.subplots(figsize=(10.0, 8.0))
axis.bar([0, 1], values, width=0.55, color=["white", "#C44E52"], edgecolor="black", linewidth=1.5)
axis.set_xticks([0, 1], ["Analytical value", "Numerical result"])
axis.set_ylabel(r"Eddy viscosity $\nu_t$ ($\mathrm{m^2\,s^{-1}}$)")
axis.tick_params(top=True, right=True, length=6, width=1.0)
axis.grid(False)
for spine in axis.spines.values():
    spine.set_linewidth(1.0)
figure.tight_layout()
figure.savefig(CASE / "验证结果.png", dpi=600, facecolor="white")
plt.close(figure)
print(f"wrote={CASE / '验证结果.png'}")
