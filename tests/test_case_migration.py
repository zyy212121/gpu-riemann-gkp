from pathlib import Path
import importlib.util, shutil, pytest
ROOT=Path(__file__).resolve().parents[1]
def module():
    spec=importlib.util.spec_from_file_location('migrate_case',ROOT/'tools/migrate_case.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_acoustic_case_preserves_thermodynamics_and_numerics(tmp_path):
    case=tmp_path/'case'
    shutil.copytree(ROOT/'legacy/public-3.0/examples/gks_flux_validation/acousticWave/N50',case)
    module().migrate(case,'L0')
    fluid=(case/'constant/fluidProperties').read_text()
    assert 'Cp 1004.5;' in fluid and 'mu 0;' in fluid and 'Pr 0.72;' in fluid
    import re
    molecular_weight=float(re.search(r'molWeight ([^;]+);',fluid)[1])
    assert abs(8314.47006650545/molecular_weight-287.0)<1e-10
    schedule=(case/'constant/schedulingProperties').read_text()
    assert 'gpuResidentPureGasOnly true;' in schedule
    assert 'gpuResearchVariant L0;' in schedule
    schemes=(case/'system/fvSchemes').read_text()
    assert 'fluxScheme Kurganov;' in schemes and 'default SSPRK2;' in schemes
    assert 'div(phi,U) Gauss MUSCL;' in schemes

def test_restart_header_conversion_preserves_every_payload_byte(tmp_path):
    restart=tmp_path/'particles'
    payload=bytes(range(256))*4
    restart.write_bytes(b'GPU3_PARTICLES_V1_BIN 7 7\n'+payload)
    module().convert_restart(restart)
    assert restart.read_bytes()==b'UGKP_PARTICLES_SCHEMA1_BIN 7 7\n'+payload

def test_unknown_state_rejected_before_case_changes(tmp_path):
    with pytest.raises(ValueError,match='state'):
        module().migrate(tmp_path,'typo')
    assert list(tmp_path.iterdir())==[]

@pytest.mark.parametrize('fault',['missing_control','bad_restart','adapter_model'])
def test_invalid_input_cannot_leave_a_half_migrated_case(tmp_path,fault):
    case=tmp_path/'case'
    shutil.copytree(ROOT/'legacy/public-3.0/examples/gks_flux_validation/acousticWave/N50',case)
    if fault=='missing_control':(case/'system/controlDict').unlink()
    elif fault=='bad_restart':
        (case/'0').mkdir(exist_ok=True)
        (case/'0/gpuResidentStrictParticles.dat').write_bytes(b'UNKNOWN 1\n')
    else:
        with (case/'constant/ugkwpProperties').open('a') as stream:stream.write('\ndragModel constantResponseTime;\n')
    before={str(p.relative_to(case)):p.read_bytes() for p in case.rglob('*') if p.is_file()}
    with pytest.raises((ValueError,FileNotFoundError)):module().migrate(case,'L0')
    after={str(p.relative_to(case)):p.read_bytes() for p in case.rglob('*') if p.is_file()}
    assert after==before
