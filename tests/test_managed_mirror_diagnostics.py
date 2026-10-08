"""A missing input must not hide unrelated drift or permit partial sync."""
from pathlib import Path
import json
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('sync', [False, True])
def test_missing_and_drift_are_both_reported_without_mutation(tmp_path, sync):
    names = ('gpu-riemann-gkp-main', 'ugkp-thermal')
    entries = [dict(path='common/'+name, upstream=names[1], downstream=names[0])
               for name in ('missing.cuh', 'changed.cuh')]
    manifest = dict(schema=1, inventory_roots=['common'], mirrors=entries,
                    local_only={name: [] for name in names})
    for name in names:
        root = tmp_path/name
        (root/'common').mkdir(parents=True)
        (root/'tools').mkdir()
        (root/'tools/managed_mirrors.json').write_text(json.dumps(manifest))
        (root/'common/changed.cuh').write_text(name)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = subprocess.run([sys.executable, str(ROOT/'tools/managed_mirrors.py'),
        '--pair-root', str(tmp_path)] + (['--sync'] if sync else []), capture_output=True, text=True)
    assert result.returncode == 1
    assert 'missing' in result.stdout and 'common/missing.cuh' in result.stdout
    assert 'mirror drift: common/changed.cuh' in result.stdout
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
