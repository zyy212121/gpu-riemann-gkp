#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def scalar_values(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    uniform = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if uniform:
        return [float(uniform.group(1))]
    block = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)\s*;",
        text,
        re.DOTALL,
    )
    if not block:
        raise RuntimeError(f"cannot parse scalar field {path}")
    return [float(value) for value in block.group(1).split()]


def main() -> None:
    log = (ROOT / "log.solver").read_text(encoding="utf-8", errors="replace")
    if "GPU2.8: separated-backend explicit GPU turbulence solver" not in log:
        raise RuntimeError("new solver identity is absent from log")
    if "wallTreatment=wallFunction" not in log:
        raise RuntimeError("SST wall-function mode is absent from log")
    if "FOAM FATAL" in log or "CUDA" in log and "failed:" in log:
        raise RuntimeError("solver log contains a fatal error")

    times = sorted(
        (
            path
            for path in ROOT.iterdir()
            if path.is_dir() and re.fullmatch(r"\d+(?:\.\d+)?(?:e[-+]?\d+)?", path.name)
        ),
        key=lambda path: float(path.name),
    )
    latest = times[-1]
    if float(latest.name) < 2.0e-5 - 1.0e-12:
        raise RuntimeError(f"SST smoke stopped early at {latest.name}")

    result: dict[str, object] = {"time": float(latest.name)}
    for name, lower_bound in (("k", 0.0), ("omega", 0.0), ("nut", 0.0)):
        values = scalar_values(latest / name)
        if not values or any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite {name}")
        if min(values) < lower_bound:
            raise RuntimeError(f"negative {name}")
        result[name] = {
            "minimum": min(values),
            "maximum": max(values),
        }
    if result["k"]["minimum"] < 1.0e-12:  # type: ignore[index]
        raise RuntimeError("k floor was violated")
    if result["omega"]["minimum"] < 1.0e-6:  # type: ignore[index]
        raise RuntimeError("omega floor was violated")
    if result["nut"]["maximum"] <= 0.0:  # type: ignore[index]
        raise RuntimeError("SST did not produce positive eddy viscosity")

    temperatures = scalar_values(latest / "T")
    if not temperatures or any(not math.isfinite(value) for value in temperatures):
        raise RuntimeError("non-finite gas temperature")
    mean_temperature = sum(temperatures)/len(temperatures)
    if mean_temperature >= 1200.0:
        raise RuntimeError("wall-function heat flux did not cool the gas")
    result["T"] = {
        "minimum": min(temperatures),
        "maximum": max(temperatures),
        "mean": mean_temperature,
        "initial": 1200.0,
    }

    (ROOT / "check_wall_function_gpu.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
