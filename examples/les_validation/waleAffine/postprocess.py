#!/usr/bin/env python3
"""Generate the affine-flow eddy-viscosity validation plot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE = Path(__file__).resolve().parent
reports = [
    json.loads((CASE / "check_trace_free.json").read_text(encoding="utf-8")),
    json.loads((CASE / "check_compressible.json").read_text(encoding="utf-8")),
]
with (CASE / "validation_data.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(
        (
            "mode",
            "divergence_1_s",
            "nut_exact_m2_s",
            "nut_numerical_m2_s",
            "relative_error",
        )
    )
    for report in reports:
        writer.writerow(
            (
                report["mode"],
                report["divergence"],
                report["expectedNut"],
                report["nut"],
                report["relativeError"],
            )
        )

plt.rcParams.update({"font.family": "serif", "font.size": 18, "axes.labelsize": 22, "xtick.labelsize": 18, "ytick.labelsize": 18, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "mathtext.fontset": "stix", "axes.unicode_minus": False})
with (CASE / "validation_data.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
exact = [float(row["nut_exact_m2_s"]) for row in rows]
numerical = [float(row["nut_numerical_m2_s"]) for row in rows]
figure, axis = plt.subplots(figsize=(10.0, 8.0))
positions = [0.0, 1.0]
width = 0.34
axis.bar([value - width / 2 for value in positions], exact, width=width, color="white", edgecolor="black", linewidth=1.5, label="OpenFOAM analytical")
axis.bar([value + width / 2 for value in positions], numerical, width=width, color="#C44E52", edgecolor="black", linewidth=1.5, label="GPU-Riemann-GKP")
axis.set_xticks(positions, ["Trace-free", "Compressible"])
axis.set_ylabel(r"Eddy viscosity $\nu_t$ ($\mathrm{m^2\,s^{-1}}$)")
axis.tick_params(top=True, right=True, length=6, width=1.0)
axis.grid(False)
for spine in axis.spines.values():
    spine.set_linewidth(1.0)
axis.legend(frameon=False)
figure.tight_layout()
figure.savefig(CASE / "验证结果.png", dpi=600, facecolor="white")
plt.close(figure)
print(f"wrote={CASE / '验证结果.png'}")
