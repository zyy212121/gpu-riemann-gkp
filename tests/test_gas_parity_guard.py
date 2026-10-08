"""Precise parity exceptions must never suppress unrelated production drift."""
from pathlib import Path
import json
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]


def check(tmp_path,path,candidate,reference):
    for label,text in [('candidate',candidate),('reference',reference)]:
        p=tmp_path/label/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    return subprocess.run([sys.executable,str(ROOT/'tools/check_gas_parity.py'),
        '--candidate',str(tmp_path/'candidate'),'--reference',str(tmp_path/'reference')],capture_output=True,text=True)


def test_equal_formal_configuration_passes(tmp_path):
    q=check(tmp_path,'common/GpuSchedulingConfiguration.H','int shared=1;\n','int shared=1;\n')
    assert q.returncode==0,q.stdout+q.stderr

@pytest.mark.parametrize('field',['heavyReductionAutoInterval','particleCapacity','maxFaceWalkHops'])
def test_nonresearch_configuration_drift_is_rejected(tmp_path,field):
    q=check(tmp_path,'common/GpuSchedulingConfiguration.H',f'int {field}=2;\n',f'int {field}=1;\n')
    assert q.returncode==1 and 'GpuSchedulingConfiguration.H' in q.stdout,q.stdout+q.stderr


def test_only_exact_research_include_is_exempt(tmp_path):
    text='#include "GpuResearchHierarchy.H"\nint shared=1;\n'
    q=check(tmp_path,'common/GpuSchedulingConfiguration.H',text,'int shared=1;\n')
    assert q.returncode==0,q.stdout+q.stderr
    q=check(tmp_path,'common/GpuSchedulingConfiguration.H',text.replace('shared=1','shared=2'),'int shared=1;\n')
    assert q.returncode==1,q.stdout+q.stderr


def test_only_inactive_thermal_transport_branch_is_exempt(tmp_path):
    text='#if GPU_OPERATOR_THERMAL\nint thermal=1;\n#endif\nint gas=1;\n'
    q=check(tmp_path,'common/GpuParticleTransport.cuh',text.replace('thermal=1','thermal=2'),text)
    assert q.returncode==0,q.stdout+q.stderr
    q=check(tmp_path,'common/GpuParticleTransport.cuh',text.replace('gas=1','gas=2'),text)
    assert q.returncode==1,q.stdout+q.stderr


def test_missing_shared_input_and_other_drift_are_both_reported(tmp_path):
    q=check(tmp_path,'common/Shared.cuh','int gas=2;\n','int gas=1;\n')
    p=tmp_path/'reference/common/Missing.cuh';p.write_text('int missing;\n')
    for label in ['candidate','reference']:
        p=tmp_path/label/'tools/managed_mirrors.json';p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps({'mirrors':[{'path':'common/Missing.cuh'}]}))
    q=subprocess.run([sys.executable,str(ROOT/'tools/check_gas_parity.py'),'--candidate',str(tmp_path/'candidate'),'--reference',str(tmp_path/'reference')],capture_output=True,text=True)
    assert q.returncode==1 and 'Missing.cuh' in q.stdout and 'Shared.cuh' in q.stdout,q.stdout+q.stderr
