from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
def test_activation_isolates_binary_and_backend_from_engineering_install():
    result = subprocess.run(['bash', '-c', 'export WM_PROJECT_VERSION=10 WM_PROJECT_DIR=/opt/openfoam10 FOAM_USER_APPBIN=/tmp/engineering-bin; source "$1"; printf "%s\n%s\n" "$FOAM_USER_APPBIN" "$GAS_UGKP_CUDA_BACKEND"', 'bash', str(ROOT/'scripts/activate.sh')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(ROOT/'build/bin'), str(ROOT/'build/bin/gasUGKPCudaBackend')]

def test_unbuilt_package_never_falls_back_to_engineering_solver(tmp_path):
    import shutil,os
    for rel in ['scripts/activate.sh','scripts/openfoam10-env.sh','scripts/bin/gasUGKP']:
        dst=tmp_path/'package'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if (ROOT/rel).exists():shutil.copy2(ROOT/rel,dst)
    engineering=tmp_path/'engineering';engineering.mkdir()
    executable=engineering/'gasUGKP';executable.write_text('#!/bin/sh\necho ENGINEERING\n');executable.chmod(0o755)
    env=dict(os.environ,PATH=str(engineering)+':'+os.environ['PATH'],WM_PROJECT_VERSION='10',WM_PROJECT_DIR='/opt/openfoam10',FOAM_USER_APPBIN=str(engineering))
    result=subprocess.run(['bash','-c','source "$1"; gasUGKP', 'bash',str(tmp_path/'package/scripts/activate.sh')],env=env,capture_output=True,text=True)
    assert result.returncode==127
    assert 'ENGINEERING' not in result.stdout
