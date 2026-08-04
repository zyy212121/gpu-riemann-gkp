#!/usr/bin/env python3
"""Compare the interior-cell nut against the analytic affine-gradient value."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


CASE = Path(__file__).resolve().parent
EXPECTED = 0.07079025356643386


def latest_time() -> Path:
    times = []
    for path in CASE.iterdir():
        try:
            value = float(path.name)
        except ValueError:
            continue
        if path.is_dir() and value > 0.0:
            times.append((value, path))
    if not times:
        raise RuntimeError("no result time directory was written")
    return max(times)[1]


def scalar_values(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if match:
        values = [float(value) for value in match.group(2).split()]
        if len(values) != int(match.group(1)):
            raise RuntimeError("invalid nut list length")
        return values
    match = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if not match:
        raise RuntimeError("cannot parse nut internalField")
    return [float(match.group(1))] * 27


time_dir = latest_time()
nut = scalar_values(time_dir / "nut")
if len(nut) != 27:
    raise RuntimeError("expected 27 cells")
centre = nut[13]
relative_error = abs(centre - EXPECTED) / EXPECTED
if not math.isfinite(centre) or centre < 0.0:
    raise RuntimeError("central nut is non-finite or negative")
if relative_error > 1.0e-6:
    raise RuntimeError(
        f"Smagorinsky affine check failed: value={centre:.16g}, "
        f"expected={EXPECTED:.16g}, relativeError={relative_error:.3e}"
    )

report = {
    "time": time_dir.name,
    "model": "Smagorinsky",
    "centralCell": 13,
    "nut": centre,
    "expectedNut": EXPECTED,
    "relativeError": relative_error,
}
(CASE / "check_affine_nut.json").write_text(json.dumps(report, indent=2) + "\n")
with (CASE / "validation_data.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(("model", "nut_exact_m2_s", "nut_numerical_m2_s", "relative_error"))
    writer.writerow((report["model"], EXPECTED, centre, relative_error))
print(json.dumps(report, indent=2))
