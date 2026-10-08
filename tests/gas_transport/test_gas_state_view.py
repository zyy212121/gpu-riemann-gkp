"""Actual shared device headers on a serial host emulator, not a CUDA runtime test."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("adapter", ["gas_view", "legacy", "legacy_optional_species"])
def test_gas_only_view_runs_every_legacy_operator_without_application_state(tmp_path, bits, adapter):
    assert (ROOT / "common/gasTransport/GasStateView.H").is_file(), "missing independent gas-only state view"
    exe = tmp_path / f"gas{bits}"
    flags = ["-DLEGACY_REFERENCE"] if adapter.startswith("legacy") else []
    if adapter == "legacy_optional_species": flags.append("-DLEGACY_OPTIONAL_SPECIES")
    build = subprocess.run(["g++", *flags, "-std=c++17", "-O0", f"-DUGKWP_GPU_REAL_BITS={bits}", "-I"+str(ROOT / "common"), "-I"+str(ROOT / "common/gpu"), "-I"+str(ROOT / "common/gasNumerics"), str(HERE / "gas_state_probe.cpp"), "-o", str(exe)], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout == (HERE / f"thermal_aligned_gas_operator_fp{bits}.txt").read_text()

@pytest.mark.parametrize("bits", [32, 64])
def test_approved_sst_alignment_preserves_all_non_sst_legacy_rows(bits):
    import hashlib, json
    old=(HERE/f"legacy_gas_operator_fp{bits}.txt").read_bytes()
    new=(HERE/f"thermal_aligned_gas_operator_fp{bits}.txt").read_bytes()
    proof=json.loads((HERE/"thermal_alignment_provenance.json").read_text())
    row=next(r for r in proof["matrix"] if r["real_bits"]==bits)
    assert hashlib.sha256(old).hexdigest()==row["legacy_sha256"]
    assert hashlib.sha256(new).hexdigest()==row["aligned_sha256"]
    before,after=old.decode().splitlines(),new.decode().splitlines()
    assert len(before)==len(after)==324
    changed=[(a,b) for a,b in zip(before,after) if a!=b]
    assert len(changed)==81
    assert all(a.split()[:4]==b.split()[:4] and a.split()[3]=="3" for a,b in changed)
