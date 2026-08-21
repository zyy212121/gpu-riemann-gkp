#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
EVIDENCE = PACKAGE.parent
BASE = EVIDENCE / "05_heavy_laval_nozzle" / "weighted_tail_s1_s2"
NEW = PACKAGE / "source_calculation_outputs"
FACTORS = (1, 2, 4, 8, 16, 32, 64)
BOOTSTRAP_SAMPLES = 200_000
BOOTSTRAP_SEED = 20260820


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


base_stage = {
    int(row["factor"]): row
    for row in read_rows(BASE / "gpu30_heavy_tail_stage_cost.csv")
}
base_speed = {
    int(row["factor"]): row
    for row in read_rows(BASE / "gpu30_heavy_tail_speedup.csv")
}
base_process = read_rows(BASE / "authoritative_summaries" / "process_medians.csv")
new_process = [
    row for row in read_rows(NEW / "process_medians.csv")
    if int(row["measured"]) == 1
]

rng = np.random.default_rng(BOOTSTRAP_SEED)
stage_rows = []
speed_rows = []

for factor in FACTORS:
    old = base_stage[factor]
    new_group = [row for row in new_process if int(row["factor"]) == factor]
    if len(new_group) != 5:
        raise RuntimeError(f"factor {factor}: expected 5 measured newS2 runs, got {len(new_group)}")

    new_total = np.array([float(row["total_ms"]) for row in new_group])
    new_pool = np.array([float(row["collision_pool_ms"]) for row in new_group])
    new_moments = np.array([float(row["moments_ms"]) for row in new_group])
    new_total_median = float(np.median(new_total))

    stage_rows.append({
        "factor": factor,
        "initial_refined_cell_parcels": old["initial_refined_cell_parcels"],
        "initial_total_parcels": old["initial_total_parcels"],
        "measured_median_max_cell_parcels": old["measured_median_max_cell_parcels"],
        "collision_pool_s1_ms": old["collision_pool_s1_ms"],
        "collision_pool_s2_ms": old["collision_pool_s2_ms"],
        "collision_pool_newS2_ms": f"{np.median(new_pool):.12g}",
        "moments_s1_ms": old["moments_s1_ms"],
        "moments_s2_ms": old["moments_s2_ms"],
        "moments_newS2_ms": f"{np.median(new_moments):.12g}",
        "total_s1_ms": old["total_s1_ms"],
        "total_s2_ms": old["total_s2_ms"],
        "total_newS2_ms": f"{new_total_median:.12g}",
    })

    s1 = np.array([
        float(row["total_ms"]) for row in base_process
        if int(row["factor"]) == factor and row["state"] == "S1"
    ])
    s2 = np.array([
        float(row["total_ms"]) for row in base_process
        if int(row["factor"]) == factor and row["state"] == "S2"
    ])
    if len(s1) != 5 or len(s2) != 5:
        raise RuntimeError(f"factor {factor}: expected 5 S1 and 5 S2 measured baseline runs")

    s1_boot = np.median(rng.choice(s1, size=(BOOTSTRAP_SAMPLES, len(s1))), axis=1)
    s2_boot = np.median(rng.choice(s2, size=(BOOTSTRAP_SAMPLES, len(s2))), axis=1)
    new_boot = np.median(
        rng.choice(new_total, size=(BOOTSTRAP_SAMPLES, len(new_total))), axis=1
    )
    s1_new_ci = np.quantile(s1_boot / new_boot, (0.025, 0.975))
    s2_new_ci = np.quantile(s2_boot / new_boot, (0.025, 0.975))
    published = base_speed[factor]

    speed_rows.append({
        "factor": factor,
        "initial_refined_cell_parcels": old["initial_refined_cell_parcels"],
        "initial_total_parcels": old["initial_total_parcels"],
        "measured_median_max_cell_parcels": old["measured_median_max_cell_parcels"],
        "s1_over_s2": published["median_speedup"],
        "s1_over_s2_ci95_low": published["ci95_low"],
        "s1_over_s2_ci95_high": published["ci95_high"],
        "s1_over_newS2": f"{float(np.median(s1)) / new_total_median:.16g}",
        "s1_over_newS2_ci95_low": f"{s1_new_ci[0]:.16g}",
        "s1_over_newS2_ci95_high": f"{s1_new_ci[1]:.16g}",
        "s2_over_newS2": f"{float(np.median(s2)) / new_total_median:.16g}",
        "s2_over_newS2_ci95_low": f"{s2_new_ci[0]:.16g}",
        "s2_over_newS2_ci95_high": f"{s2_new_ci[1]:.16g}",
    })

write_rows(
    HERE / "gpu30_heavy_tail_stage_cost.csv",
    list(stage_rows[0]),
    stage_rows,
)
write_rows(
    HERE / "gpu30_heavy_tail_speedup.csv",
    list(speed_rows[0]),
    speed_rows,
)
