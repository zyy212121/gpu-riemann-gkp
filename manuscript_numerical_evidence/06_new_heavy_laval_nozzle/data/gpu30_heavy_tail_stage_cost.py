#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DATA = HERE / "gpu30_heavy_tail_stage_cost.csv"
OUTPUT = HERE.parent / "figures" / "gpu30_heavy_tail_stage_cost.png"

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

styles = {
    "S1": dict(color="#C44E52", linestyle="-", marker="o"),
    "S2": dict(color="#4C72B0", linestyle="--", marker="s"),
    "newS2": dict(color="#55A868", linestyle="-.", marker="^"),
}
field_suffix = {"S1": "s1", "S2": "s2", "newS2": "newS2"}

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), sharex=True)
panels = (
    ("collision_pool", "(a) Collision pool"),
    ("moments", "(b) Moment reduction"),
)

for axis, (prefix, label) in zip(axes, panels):
    for state in ("S1", "S2", "newS2"):
        values = [float(row[f"{prefix}_{field_suffix[state]}_ms"]) for row in rows]
        style = styles[state]
        axis.plot(
            x,
            values,
            label=state,
            linewidth=1.6,
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.0,
            **style,
        )
    axis.text(0.04, 0.94, label, transform=axis.transAxes, va="top", fontsize=18)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.tick_params(top=True, right=True, length=6, width=1.0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)

for axis in axes:
    axis.set_ylabel("Per-step kernel time (ms)")
axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.84))
fig.supxlabel(r"Initial refined-cell occupancy $n_{c,0}$ ($10^3$ parcels)", y=0.025)
fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.95, wspace=0.30)
fig.savefig(OUTPUT, dpi=600, facecolor="white")
plt.close(fig)
