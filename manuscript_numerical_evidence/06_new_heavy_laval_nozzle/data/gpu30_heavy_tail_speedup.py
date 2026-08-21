#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DATA = HERE / "gpu30_heavy_tail_speedup.csv"
OUTPUT = HERE.parent / "figures" / "gpu30_heavy_tail_speedup.png"

with DATA.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

labels = [str(int(float(row["initial_refined_cell_parcels"]) / 1000.0)) for row in rows]
x = list(range(len(rows)))

plt.rcParams.update({
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
})

fig, axis = plt.subplots(figsize=(10.0, 8.0))
axis.axhline(1.0, color="black", linestyle="--", linewidth=2.0, label="Break-even")

series = (
    ("s1_over_s2", "S1/S2 total-time speedup", "#C44E52", "-", "o"),
    ("s1_over_newS2", "S1/newS2 total-time speedup", "#55A868", "-.", "^"),
    ("s2_over_newS2", "S2/newS2 total-time speedup", "#4C72B0", "--", "s"),
)

for field, label, color, linestyle, marker in series:
    values = [float(row[field]) for row in rows]
    lower = [value - float(row[f"{field}_ci95_low"]) for value, row in zip(values, rows)]
    upper = [float(row[f"{field}_ci95_high"]) - value for value, row in zip(values, rows)]
    axis.errorbar(
        x,
        values,
        yerr=[lower, upper],
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=1.6,
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
        capsize=3.0,
    )

axis.set_xticks(x)
axis.set_xticklabels(labels)
axis.set_xlabel(r"Initial refined-cell occupancy $n_{c,0}$ ($10^3$ parcels)")
axis.set_ylabel("Total-time speedup")
axis.tick_params(top=True, right=True, length=6, width=1.0)
axis.grid(False)
for spine in axis.spines.values():
    spine.set_linewidth(1.0)
axis.legend(frameon=False, loc="upper left")
fig.tight_layout(pad=0.7)
fig.savefig(OUTPUT, dpi=600, facecolor="white")
plt.close(fig)
